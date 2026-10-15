"""ROS2 node: camera -> board detection -> rules engine -> robot.

Game flow
---------
1. BoardDetection runs its calibration UI (corners, piece calibration,
   difficulty).
2. Press 'S' with the pieces in the start position (or 'F' to start from
   whatever position is on the board, human to move).
3. Each side moves in turn under WCDF (English draughts) rules, BLACK (robot)
   first unless the ``human_first`` parameter is true:
   * HUMAN (white): move a piece, take your hand away.  When the detected
     board has been stable for ``stable_time`` seconds the move is matched
     against every legal move.  If you ignored a mandatory capture (or
     stopped a multi-jump early) the move is accepted but the robot "huffs"
     (removes) the piece that should have captured.  Pieces you capture you
     take off the board yourself.
   * ROBOT (black): the AI only plays legal moves (it always captures and
     completes multi-jumps).  The Move message's ``removed_pieces`` contains
     the captured pieces and, after a huff, the huffed piece.
4. The game ends when a side has no pieces / no legal move, or by draw
   (40 moves each without capture or man move, or 3-fold repetition).

Keys (any OpenCV window focused): S start, F force-start from detected
position, 1/2/3 difficulty, SPACE re-check the board now, R robot finished
(manual), H print your legal moves, Q quit.
"""

import threading
import time
from enum import Enum, auto

import cv2
import numpy as np
import pygame
import rclpy
from rclpy.node import Node

from checkers_msgs.msg import Board, HandDetected, Move, Piece, RobotMove

from checkers_game.camera.board_detection import DEBUG_MODE, BoardDetection
from checkers_game.camera.ximea_camera import XimeaCamera
from checkers_game.checkers.board import Board as BoardView
from checkers_game.checkers.game import HUMAN, ROBOT, Game
from checkers_game.checkers.rules import DRAW, color_name, square_name
from checkers_game.checkers.rules import GameState as RulesState
from checkers_game.constants import HEIGHT, WIDTH

try:  # mediapipe is optional - without it we rely on motion detection only
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None


class Difficulty(Enum):
    """AI difficulty levels (value = level for minimax.depth_for_level)."""
    EASY = 1
    MEDIUM = 3
    HARD = 5


class GameState(Enum):
    WAITING_FOR_START = auto()
    PLAYER_TURN = auto()
    ROBOT_MOVING = auto()
    GAME_OVER = auto()


def print_grid(title, grid, highlight=()):
    print(f"\n  {title}")
    print("      " + "  ".join(str(c) for c in range(8)))
    sym = {0: ".", 1: "w", 2: "b", 3: "o", -1: "?"}
    for r in range(8):
        row = []
        for c in range(8):
            ch = sym.get(int(grid[r][c]), "?")
            row.append(f"[{ch}]" if (r, c) in highlight else f" {ch} ")
        print(f"   {r} " + "".join(row))


class CheckersNode(Node):
    def __init__(self):
        super().__init__('checkers_game')
        self.declare_parameter('human_first', False)
        self.declare_parameter('ai_time_limit', 4.0)
        self.declare_parameter('stable_time', 0.8)
        self.declare_parameter('motion_threshold', 1.5)

        print("\n" + "=" * 60)
        print("INITIALIZING CHECKERS NODE")
        print("=" * 60)

        pygame.init()
        self.WIN = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Checkers")
        self.clock = pygame.time.Clock()
        self.FPS = 30

        print("Creating camera connection...")
        self.ximeaCamera = XimeaCamera()
        print("✓ Camera connected")

        print("\nInitializing board detection...")
        self.boardDetection = BoardDetection(self.ximeaCamera)
        print("✓ Board detection ready")

        self.game = Game(human_first=self.get_parameter('human_first').value)
        self.view = BoardView(dark_parity=getattr(self.boardDetection, 'dark_parity', 1))

        self.hands = None
        if mp is not None:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(max_num_hands=2, model_complexity=0)
            self.mp_drawing = mp.solutions.drawing_utils

        self.game_state = GameState.WAITING_FOR_START
        self.difficulty_level = int(getattr(self.boardDetection, 'selected_difficulty', 3))
        self.ai_time_limit = float(self.get_parameter('ai_time_limit').value)
        self.stable_time = float(self.get_parameter('stable_time').value)
        self.motion_threshold = float(self.get_parameter('motion_threshold').value)

        # robot sync (callbacks and timer run in the same executor thread)
        self.robot_done = True
        self.robot_finished_at = None
        self.verify_after_robot = False

        # detection stability tracking
        self.cam_lock = threading.Lock()
        self.latest_camera_image = None
        self.capture_running = True
        self.last_grid = None
        self.grid_stable_since = time.time()
        self.prev_small = None
        self.motion = 0.0
        self.last_motion_time = time.time()
        self.last_hand_time = 0.0
        self.HAND_COOLDOWN = 0.7
        self.last_checked_sig = None
        self.highlight = []
        self.last_move = None
        self.latest_grid = None
        self.last_image = None
        self.ref_stale = False

        self.board_publisher = self.create_publisher(Board, '/board_topic', 10)
        self.move_publisher = self.create_publisher(Move, '/move_topic', 10)
        self.hand_detected_publisher = self.create_publisher(HandDetected, '/hand_detected', 10)
        self.robot_move_subscription = self.create_subscription(
            RobotMove, '/robot_move_topic', self.robot_move_callback, 10)

        print("\n" + "=" * 60)
        print("SYSTEM READY")
        print("=" * 60)
        print("  S = start (pieces in start position)   F = start from current position")
        print("  1/2/3 = easy/medium/hard   SPACE = re-check board now")
        print("  R = robot finished (manual)   H = show my legal moves   Q = quit")
        print(f"  Rules: English draughts (WCDF) + huff rule; "
              f"{'you' if self.game.first == HUMAN else 'robot (black)'} moves first")
        print(f"  Difficulty level: {self.difficulty_level}\n")

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.timer = self.create_timer(1.0 / self.FPS, self.update)

    # ------------------------------------------------------------- camera
    def _capture_loop(self):
        while self.capture_running:
            try:
                img = self.ximeaCamera.get_camera_image()
                with self.cam_lock:
                    self.latest_camera_image = img
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"camera error: {e}")
                time.sleep(0.2)
            time.sleep(0.01)

    def _grab(self):
        with self.cam_lock:
            img = self.latest_camera_image
        if img is None:
            img = self.ximeaCamera.get_camera_image()
        return img

    # ---------------------------------------------------------- callbacks
    def robot_move_callback(self, msg):
        if not msg.robot_move_done:
            return
        if self.game_state != GameState.ROBOT_MOVING:
            return
        print("\n✓ ROBOT FINISHED")
        self._robot_finished()

    def _robot_finished(self):
        self.robot_done = True
        self.robot_finished_at = time.time()
        self.verify_after_robot = True
        self.last_checked_sig = None
        if self.game.winner() is not None:
            self._game_over()
            return
        self.game_state = GameState.PLAYER_TURN
        print("→ Your turn (WHITE). Move, then take your hand off the board.\n")

    # ------------------------------------------------------------ main loop
    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._shutdown()
                return
        image = self._grab()
        if image is None:
            return

        self.last_image = image
        expected = None
        if self.game.started and (self.game_state == GameState.ROBOT_MOVING
                                  or self.verify_after_robot):
            expected = self.game.expected_grid()
        grid = self.boardDetection.get_board(image, expected=expected)
        grid = np.asarray(grid, dtype=int)
        # follow lighting changes (lamp on/off): gentle exposure correction,
        # never while the robot or a hand is over the board
        if (self.game_state != GameState.ROBOT_MOVING
                and time.time() - self.last_hand_time > 2.0
                and self.boardDetection.last_light is not None
                and hasattr(self.ximeaCamera, 'track_exposure')):
            self.ximeaCamera.track_exposure(self.boardDetection.last_light)
        self.latest_grid = grid
        self._track_stability(image, grid)

        hand = self.detect_hand(image)
        if hand:
            self.last_hand_time = time.time()

        key = cv2.waitKey(1) & 0xFF
        self._handle_key(key, grid)

        if self.game_state == GameState.PLAYER_TURN:
            self._player_turn(grid)

        self._draw(grid)

    def _track_stability(self, image, grid):
        warped = self.boardDetection.warp(image)
        small = cv2.cvtColor(cv2.resize(warped, (160, 160), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY if warped.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        if self.prev_small is not None:
            diff = cv2.absdiff(small, self.prev_small)
            self.motion = float((diff > 25).mean() * 100)
            if self.motion > self.motion_threshold:
                self.last_motion_time = time.time()
        self.prev_small = small
        if self.last_grid is None or not np.array_equal(grid, self.last_grid):
            self.grid_stable_since = time.time()
            self.last_grid = grid.copy()

    def _board_is_settled(self):
        now = time.time()
        return (now - self.grid_stable_since >= self.stable_time
                and now - self.last_motion_time >= self.stable_time
                and now - self.last_hand_time >= self.HAND_COOLDOWN)

    def _handle_key(self, key, grid):
        if key == 255:
            return
        if key in (ord('1'), ord('2'), ord('3')):
            lvl = {ord('1'): Difficulty.EASY, ord('2'): Difficulty.MEDIUM,
                   ord('3'): Difficulty.HARD}[key]
            self.difficulty_level = lvl.value
            print(f"\n★ Difficulty: {lvl.name} (level {lvl.value})")
        elif key in (ord('s'), ord('S')):
            self._start_game(grid, force=False)
        elif key in (ord('f'), ord('F')):
            self._start_game(grid, force=True)
        elif key == 32:
            print("\n[SPACE] re-checking the board now")
            self.last_checked_sig = None
            self.grid_stable_since -= self.stable_time
            self.last_motion_time -= self.stable_time
        elif key in (ord('r'), ord('R')):
            if self.game_state == GameState.ROBOT_MOVING:
                print("\n[R] robot move marked as finished manually")
                self._robot_finished()
        elif key in (ord('h'), ord('H')):
            self._print_hints()
        elif key in (ord('q'), ord('Q')):
            self._shutdown()

    # -------------------------------------------------------------- start
    def _start_game(self, grid, force):
        if self.game_state not in (GameState.WAITING_FOR_START, GameState.GAME_OVER):
            print("Game already running")
            return
        if force:
            unsure = [(r, c) for r in range(8) for c in range(8) if int(grid[r][c]) in (3, -1)]
            if unsure:
                print("\n✗ Cannot start from this position - the camera is not sure about "
                      + ", ".join(square_name(q) for q in unsure))
                print_grid("detected board", grid, unsure)
                return
            self.game.start_from_grid(grid, turn=HUMAN)
            print("\nGAME STARTED from the detected position - your move (WHITE)")
        else:
            ok, msg, parity = Game.check_standard_start(grid)
            if not ok:
                print(f"\n✗ Cannot start: {msg}")
                print_grid("detected board", grid)
                print("  Fix the pieces / detection and press S again "
                      "(or F to start from this position anyway).")
                return
            self.game.start_standard(dark_parity=parity)
            self.view.dark_parity = parity
            print("\n" + "=" * 60)
            print("GAME STARTED - English draughts (WCDF) with huff rule")
            print("=" * 60)
        print(self.game.state.ascii())
        self.verify_after_robot = False
        self.ref_stale = False
        self.game.max_noise_squares = 1
        self.boardDetection.set_reference(self.last_image, self.game.expected_grid())
        self.publish_board_state()
        self.last_checked_sig = self._sig(grid)
        self.highlight = []
        self.last_move = None
        if self.game.turn == ROBOT:
            self.game_state = GameState.PLAYER_TURN   # placeholder until robot plans
            self._robot_turn()
        else:
            self.game_state = GameState.PLAYER_TURN
            print("→ Your move (WHITE)")

    # ------------------------------------------------------------ player
    @staticmethod
    def _sig(grid):
        return np.asarray(grid, dtype=np.int8).tobytes()

    def _player_turn(self, grid):
        if not self._board_is_settled():
            return
        sig = self._sig(grid)
        if sig == self.last_checked_sig:
            return
        self.last_checked_sig = sig

        inf = self.game.infer_human_move(grid)

        if self.verify_after_robot:
            # first settled frame after the robot finished: the board must be
            # the expected position - or that position plus one legal human
            # move (the human may already have moved)
            self.verify_after_robot = False
            if inf.kind == "no_change" and not self.game.mismatches(grid, use_color=False):
                # robot move confirmed by the camera: new reference position
                self.boardDetection.set_reference(self.last_image, self.game.expected_grid())
                self.highlight = []
                return
            if not inf.accepted:
                bad = self.game.mismatches(grid, use_color=False)
                print("\n⚠ After the robot's move the board does not match the game:")
                print("  squares: " + ", ".join(square_name(q) for q in bad))
                print_grid("detected", grid, bad)
                print_grid("expected", self.game.expected_grid(), bad)
                print("  → Fix the physical board (or the robot's placement); the board is")
                print("    re-checked automatically.")
                self.highlight = bad
                self.ref_stale = True
                # until the board is put right, don't explain the robot's
                # misplaced piece away as detection noise
                self.game.max_noise_squares = 0
                return

        if inf.kind == "no_change":
            self.highlight = []
            if self.ref_stale and not self.game.mismatches(grid, use_color=False):
                # board was put right after a robot problem
                self.boardDetection.set_reference(self.last_image, self.game.expected_grid())
                self.ref_stale = False
                self.game.max_noise_squares = 1
                print("✓ Board matches the game again")
            return
        if not inf.accepted:
            self._report_invalid(inf, grid)
            return
        if inf.kind == "huff" and inf.move.is_capture and \
                time.time() - self.grid_stable_since < 3 * self.stable_time:
            # looks like a multi-jump stopped half way - maybe the player is
            # just pausing between jumps: give them more time before huffing
            self.last_checked_sig = None
            return

        self.highlight = []
        if inf.noise_squares:
            print(f"  (ignored detection noise on {', '.join(square_name(q) for q in inf.noise_squares)})")
        self.game.apply_human(inf)
        self.boardDetection.set_reference(self.last_image, self.game.expected_grid())
        self.last_move = inf.move
        print(f"\n✓ Your move: {inf.move}")
        if inf.kind == "huff":
            print("⚠ HUFF: " + inf.reason)
            print("  → The robot will remove one of: "
                  + ", ".join(square_name(q) for q in inf.huff_squares))
        self.publish_board_state()

        # with a huff pending the robot first removes a piece, which can
        # change the outcome - _robot_turn() handles "robot has no move"
        if not self.game.pending_huff and self.game.winner() is not None:
            self._game_over()
            return
        self._robot_turn()

    def _report_invalid(self, inf, grid):
        print("\n" + "=" * 60)
        print("✗ MOVE NOT ACCEPTED")
        print("=" * 60)
        print("  " + inf.reason)
        exp = self.game.expected_grid()
        print_grid("game position (before your move)", exp, inf.changed)
        print_grid("camera sees", grid, inf.changed)
        if inf.closest is not None and len(inf.closest_diff) <= 2:
            print(f"  Closest legal move: {inf.closest} - differs on "
                  + ", ".join(square_name(q) for q in inf.closest_diff))
            if inf.closest.captured and any(q in inf.closest.captured for q in inf.closest_diff):
                print("  → Did you forget to take the captured piece off the board?")
        self._print_hints()
        print("  Put the board right; it is re-checked automatically (or press SPACE).")
        self.highlight = inf.changed

    def _print_hints(self):
        moves = self.game.hint_moves()
        if moves:
            caps = self.game.state.all_captures()
            print(f"  Your legal moves{' (CAPTURE IS MANDATORY)' if caps else ''}:")
            for m in moves:
                print(f"    {m}")

    # ------------------------------------------------------------- robot
    def _robot_turn(self):
        print("\n" + "-" * 60)
        print(f"Robot thinking (level {self.difficulty_level}) ...")
        t0 = time.time()
        plan = self.game.plan_robot_turn(level=self.difficulty_level,
                                         time_limit=self.ai_time_limit)
        if plan.huffed is not None:
            print(f"  Robot huffs your piece on {square_name(plan.huffed)}")
        if plan.move is None:
            self.game.apply_robot(plan)
            self.publish_board_state()
            if plan.huffed is not None:
                print(f"  → Please take your huffed piece off {square_name(plan.huffed)} yourself "
                      "(the robot has no move left, so it does not move).")
            self._game_over()
            return
        print(f"  Robot plays {plan.move}  ({time.time() - t0:.1f}s)")
        self.game.apply_robot(plan)
        self.last_move = plan.move
        self.publish_move(plan)
        self.publish_board_state()
        self.robot_done = False
        self.game_state = GameState.ROBOT_MOVING
        print("  Waiting for the robot ... (press R if it finished but did not report)")
        print("-" * 60)

    def _game_over(self):
        self.game_state = GameState.GAME_OVER
        w = self.game.winner()
        print("\n" + "=" * 60)
        if w == DRAW:
            print("GAME OVER - DRAW")
        else:
            print(f"GAME OVER - {color_name(w)} WINS ({'you' if w == HUMAN else 'robot'})")
        print("=" * 60)
        print("Moves:\n  " + "\n  ".join(self.game.log))
        print("\nSet up the start position and press S for a new game.\n")

    # ------------------------------------------------------------ hand
    def detect_hand(self, image):
        if self.hands is None:
            return False
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB if image.shape[2] == 4 else cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return False
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = self.boardDetection.board_bbox(margin=40)
        over = False
        for lm in results.multi_hand_landmarks:
            xs = [p.x * w for p in lm.landmark]
            ys = [p.y * h for p in lm.landmark]
            if min(xs) < x1 and max(xs) > x0 and min(ys) < y1 and max(ys) > y0:
                over = True
            if DEBUG_MODE:
                self.mp_drawing.draw_landmarks(rgb, lm, self.mp_hands.HAND_CONNECTIONS)
        if DEBUG_MODE:
            cv2.rectangle(rgb, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
            cv2.imshow("Hand Tracking", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return over

    # ------------------------------------------------------------ drawing
    def _draw(self, grid):
        if self.game.started:
            self.view.draw(self.WIN, self.game.state, last_move=self.last_move,
                           highlight=self.highlight, detected=grid)
        else:
            # before the game starts show the raw detection
            self.view.draw(self.WIN, RulesState.from_grid(grid))
        pygame.display.set_caption(self._caption())
        pygame.display.update()

    def _caption(self):
        if self.game_state == GameState.WAITING_FOR_START:
            return "Checkers - press S to start"
        if self.game_state == GameState.GAME_OVER:
            return f"Checkers - GAME OVER: {self.game.winner_text()}"
        if self.game_state == GameState.ROBOT_MOVING:
            return "Checkers - robot moving"
        return "Checkers - your move (white)"

    # ----------------------------------------------------------- messages
    @staticmethod
    def _piece_msg(r, c, value):
        p = Piece()
        p.row, p.col = int(r), int(c)
        p.color = "white" if value > 0 else "red"
        p.king = abs(value) == 2
        return p

    def publish_board_state(self):
        msg = Board()
        for r, c, color, king in self.game.pieces_for_msg():
            p = Piece()
            p.row, p.col, p.color, p.king = int(r), int(c), color, bool(king)
            msg.pieces.append(p)
        self.board_publisher.publish(msg)

    def publish_move(self, plan):
        msg = Move()
        msg.target_row, msg.target_col = int(plan.move.end[0]), int(plan.move.end[1])
        msg.piece_for_moving = self._piece_msg(*plan.move.start, plan.piece_value)
        # huffed piece first, then the captured ones - robot_control removes
        # them before moving its own piece (a huffed square may be the target)
        for r, c, v in plan.removed:
            msg.removed_pieces.append(self._piece_msg(r, c, v))
        self.move_publisher.publish(msg)
        print(f"  → /move_topic: ({plan.move.start[0]},{plan.move.start[1]}) -> "
              f"({msg.target_row},{msg.target_col}), remove "
              f"{[(p.row, p.col) for p in msg.removed_pieces]}")

    def publish_hand_detected_state(self):
        msg = HandDetected()
        msg.hand_detected = True
        # self.hand_detected_publisher.publish(msg)   # disabled as before

    def _shutdown(self):
        self.capture_running = False
        try:
            pygame.quit()
        finally:
            rclpy.try_shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CheckersNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.capture_running = False
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
