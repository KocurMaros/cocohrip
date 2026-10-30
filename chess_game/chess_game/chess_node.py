"""ROS2 node: camera -> piece classifier -> rules (python-chess) -> engine
-> robot.

Game flow
---------
1. The board is found in the camera image (detected, cached in
   ~/.cocohrip/chess_board.json, or clicked: a1, a8, h8, h1).
2. Set up the pieces and press S.  The colour on your side (the printed
   rank-1 side, opposite the robot) is yours; with black on your side the
   robot (white) moves first.  F starts from whatever position is on the
   board (C switches your colour for that, you move first).
3. YOUR MOVE: move, take your hand away.  When the board has been still
   for ``stable_time`` the camera's probabilities for all 64 squares are
   matched against every legal move.  Captured robot pieces you take off
   yourself.  A pawn reaching the last rank stays a pawn on the board; in
   the game it is a queen (P cycles the piece) - the dashboard marks it.
4. ROBOT: the engine picks a move, robot_control executes the pick/place
   steps (chess_msgs/ChessRobotMove on /chess/robot_move: captured piece
   to the storage beside the board first, castling = king then rook) and
   answers on /robot_move_topic.  LOCKED control (default) waits for that
   answer or R; UNLOCKED (L) also continues when the camera sees the
   expected position.
5. Checkmate, stalemate, insufficient material, 50 moves, 3-fold
   repetition end the game.

Keys (window focused): S start, F start from this position, C your colour
for F, 1-5 engine level, P promotion piece, SPACE re-check now, R robot
finished, L lock/unlock, H your legal moves, Q quit.
"""

import os
import threading
import time
from enum import Enum, auto

import chess
import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from checkers_msgs.msg import RobotMove
from chess_msgs.msg import ChessRobotMove, ChessState, PickPlace

from chess_game.engines import create_engine
from chess_game.frames import OffBoardLayout, logical_to_printed
from chess_game.game import CLASSES, ChessGame
from chess_game.robot_steps import load_pieces, robot_geometry
from chess_game.ui.diagram import draw_diagram
from chess_game.vision import board_setup
from chess_game.vision.board_model import BoardModel, load_intrinsics
from chess_game.vision.camera import ChessCamera, to_display, white_level
from chess_game.vision.classifier import SquareClassifier, work_image

try:
    from checkers_game.ui.dashboard import Dashboard
except ImportError:  # pragma: no cover
    Dashboard = None

try:  # mediapipe is optional - without it we rely on motion detection only
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None

WINDOW = "Chess"
CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision", "ximea_calibration_600.npz")
PROMOTION_CYCLE = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)


class State(Enum):
    WAITING_FOR_START = auto()
    HUMAN_TURN = auto()
    ROBOT_THINKING = auto()
    ROBOT_MOVING = auto()
    GAME_OVER = auto()


class ChessNode(Node):
    def __init__(self):
        super().__init__('chess_game')
        p = self.declare_parameter
        p('engine', 'stockfish')            # stockfish | leela | maia | lc0:weights=... | builtin
        p('level', 3)                       # 1..5
        p('ai_time_limit', 2.0)
        p('model_path', '~/chess_models/latest/piece_classifier.onnx')
        p('pieces_yaml', '')                # '' = the package's config/pieces.yaml
        p('stable_time', 0.8)
        p('motion_threshold', 1.0)          # % changed pixels that count as motion
        p('a1_corner', 'BR')                # raw image corner of printed a1
        p('storage_left_col', -1.8)
        p('storage_right_col', 8.8)
        p('storage_rows', 8)
        g = self.get_parameter

        self.level = int(g('level').value)
        self.ai_time_limit = float(g('ai_time_limit').value)
        self.stable_time = float(g('stable_time').value)
        self.motion_threshold = float(g('motion_threshold').value)
        self.layout = OffBoardLayout(g('storage_left_col').value, g('storage_right_col').value,
                                     g('storage_rows').value)
        self.pieces, self.travel_margin = load_pieces(g('pieces_yaml').value or None)

        print("Opening the camera ...")
        self.camera = ChessCamera()
        frame = self.camera.grab()
        self.work_shape = work_image(frame).shape
        K, dist = load_intrinsics(CALIB, (self.work_shape[1], self.work_shape[0]))
        corners, a1, how = board_setup.locate(work_image(frame), g('a1_corner').value)
        print(f"Board: {how}")
        if corners is None:
            res = board_setup.click_corners(WINDOW, work_image(frame))
            if res is None:
                raise SystemExit("no board corners")
            corners, a1 = res
            board_setup.save_cached(corners, a1, self.work_shape)
        self.model = BoardModel(corners, K, dist, a1)
        self.board_mask = np.zeros(self.work_shape[:2], np.uint8)
        cv2.fillConvexPoly(self.board_mask, self.model.corners.astype(np.int32), 1)
        self.board_mask = self.board_mask.astype(bool)
        exp, level = self.camera.auto_expose(
            measure=lambda f: white_level(work_image(f), self.board_mask))
        print(f"Camera exposure {exp} us (board white level {level:.0f})")

        model_path = os.path.expanduser(g('model_path').value)
        try:
            self.classifier = SquareClassifier(model_path)
            print(f"Piece classifier: {model_path}")
        except FileNotFoundError as e:
            self.classifier = None
            print(f"✗ {e}\n  The camera view works, but no game can be played without it.")
        self.engine = create_engine(g('engine').value)
        print(f"Engine: {self.engine.name}, level {self.level}")

        self.game = ChessGame()
        self.human_white_for_f = True
        self.state = State.WAITING_FOR_START
        self.control_locked = True
        self.verify_after_robot = False
        self.pending_move = None            # engine result (set by the thinking thread)
        self.last_move = None
        self.highlight = []                 # printed squares outlined red on the game board

        # camera / stability
        self.lock = threading.Lock()
        self.latest = None
        self.capturing = True
        self.prev_small = None
        self.motion = 0.0
        self.last_motion = time.time()
        self.last_hand = 0.0
        self.obs = None                     # (64, 13) of the last settled board
        self.obs_time = 0.0
        self.force_check = False
        self.hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=0) if mp else None

        self.banner = ("Set up the pieces (your colour on your side), then press S", "info")
        self.details = []
        self.dashboard = Dashboard(show=False) if Dashboard else None

        self.move_pub = self.create_publisher(ChessRobotMove, '/chess/robot_move', 10)
        self.state_pub = self.create_publisher(ChessState, '/chess/state', 10)
        self.create_subscription(RobotMove, '/robot_move_topic', self.robot_done_callback, 10)
        threading.Thread(target=self._capture_loop, daemon=True).start()
        self.timer = self.create_timer(1.0 / 20, self.update)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW, 1320, 780)
        self.window_open = False
        self.last_exposure_check = 0.0
        print("Keys: S start  F start from this position  C colour for F  1-5 level  "
              "P promotion piece\n      SPACE re-check  R robot done  L lock/unlock  H hints  Q quit")

    # ------------------------------------------------------------ camera
    def _capture_loop(self):
        while self.capturing:
            try:
                frame = self.camera.grab()
                with self.lock:
                    self.latest = frame
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"camera: {e}")
                time.sleep(0.2)

    def _settled(self):
        now = time.time()
        return (now - self.last_motion >= self.stable_time
                and now - self.last_hand >= max(0.7, self.stable_time))

    def _track_motion(self, work):
        small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(work, (200, 167)), cv2.COLOR_BGR2GRAY),
                                 (5, 5), 0)
        if self.prev_small is not None:
            self.motion = float((cv2.absdiff(small, self.prev_small) > 25).mean() * 100)
            if self.motion > self.motion_threshold:
                self.last_motion = time.time()
        self.prev_small = small

    # ------------------------------------------------------------ main loop
    def update(self):
        if self.window_open and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            self._shutdown()                    # window closed
            return
        with self.lock:
            frame = self.latest
        if frame is None:
            return
        work = work_image(frame)
        self._track_motion(work)
        if self._settled() and self.state != State.ROBOT_MOVING:
            self._track_exposure(work)
        hand, cam_tile = self._hands(work)
        if hand:
            self.last_hand = time.time()

        new_obs = False
        if self.classifier is not None and self._settled() and (
                self.obs_time < self.last_motion or self.force_check):
            self.force_check = False
            self.obs = self.classifier.classify(work, self.model)
            self.obs_time = time.time()
            new_obs = True

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            self._key(key)

        if self.state == State.ROBOT_THINKING and self.pending_move is not None:
            self._send_robot_move(self.pending_move)
        elif new_obs and self.state == State.HUMAN_TURN:
            self._human_turn()
        elif new_obs and self.state == State.ROBOT_MOVING and not self.control_locked:
            if not self.game.mismatches(self.obs):
                print("✓ Robot finished (camera sees the expected position)")
                self._robot_finished()
        self._draw(work, cam_tile)

    def _track_exposure(self, work, every=2.0):
        """Follow lighting changes (lamp on/off): gentle correction when the
        board's white squares leave [140, 235]."""
        if time.time() - self.last_exposure_check < every:
            return
        self.last_exposure_check = time.time()
        level = white_level(work, self.board_mask)
        if 140 <= level <= 235:
            return
        old = (self.camera.exposure_us, self.camera.gain_db)
        self.camera.adjust(float(np.clip(190.0 / max(level, 1.0), 0.5, 2.0)))
        new = (self.camera.exposure_us, self.camera.gain_db)
        if new != old:
            print(f"[camera] lighting changed (white level {level:.0f}): exposure {old[0]} us "
                  f"+{old[1]:.0f} dB -> {new[0]} us +{new[1]:.0f} dB")
            self.last_motion = time.time()      # re-read the board with the new exposure

    # ------------------------------------------------------------ keys
    def _key(self, key):
        k = chr(key).lower() if key < 128 else ""
        if k and k in "12345":
            self.level = int(k)
            self.details = [f"Engine level {self.level}"]
        elif k == "s":
            self._start(from_position=False)
        elif k == "f":
            self._start(from_position=True)
        elif k == "c" and self.state in (State.WAITING_FOR_START, State.GAME_OVER):
            self.human_white_for_f = not self.human_white_for_f
            self.details = [f"F starts with you as {'WHITE' if self.human_white_for_f else 'BLACK'}"]
        elif k == "p":
            i = PROMOTION_CYCLE.index(self.game.promotion_piece)
            self.game.promotion_piece = PROMOTION_CYCLE[(i + 1) % len(PROMOTION_CYCLE)]
            self.details = [f"Your pawns promote to a {chess.piece_name(self.game.promotion_piece)}"]
        elif key == 32:
            self.force_check = True
            self.last_motion -= self.stable_time
        elif k == "r" and self.state == State.ROBOT_MOVING:
            print("[R] robot move marked as finished")
            self._robot_finished()
        elif k == "l":
            self.control_locked = not self.control_locked
            self.details = ["Control " + ("LOCKED - R when the robot is done" if self.control_locked
                                          else "UNLOCKED - continues when the board matches")]
        elif k == "h" and self.game.human_to_move:
            sans = [self.game.board.san(m) for m in self.game.board.legal_moves]
            self.details = ["Your legal moves:", " ".join(sans)]
        elif k == "q":
            self._shutdown()

    # ------------------------------------------------------------ start
    def _start(self, from_position):
        if self.state not in (State.WAITING_FOR_START, State.GAME_OVER):
            return
        if self.obs is None:
            self._ui("The camera has not read the board yet - keep your hands off", "warn")
            return
        if from_position:
            self.game.human_white = self.human_white_for_f
            ok, msg = self.game.start_from_observation(self.obs, human_to_move=True)
            if not ok:
                self._ui("Cannot start from this position", "error", [msg])
                return
        else:
            check = ChessGame.check_setup(self.obs)
            if not check.ok:
                self.highlight = check.wrong
                self._ui("Not the start position - fix it and press S", "error",
                         [check.message, "Or press F to start from the position on the board."])
                return
            self.game.start(human_white=check.human_white)
        self.highlight = []
        self.last_move = None
        self.verify_after_robot = False
        side = "WHITE" if self.game.human_white else "BLACK"
        print(f"\nGAME STARTED - you play {side}, engine {self.engine.name} level {self.level}")
        self._publish_state()
        if self.game.human_to_move:
            self.state = State.HUMAN_TURN
            self._your_turn([f"Game started - you play {side}."])
        else:
            self._robot_turn()

    # ------------------------------------------------------------ human
    def _human_turn(self):
        inf = self.game.infer_human_move(self.obs)
        if self.verify_after_robot:
            self.verify_after_robot = False
            if inf.kind == "no_change":
                self.highlight = []
                return
            if not inf.accepted:
                bad = self.game.mismatches(self.obs)
                self.highlight = bad
                self._ui("Board does not match after the robot's move", "warn",
                         ["Check the squares outlined red: " + _names(bad),
                          "Put the pieces where the game board shows them."])
                return
        if inf.kind == "no_change":
            if self.highlight or self.banner[1] == "error":
                self.highlight = []
                self._your_turn(["Board matches the game again."])
            return
        if inf.kind == "unsure":
            self._ui("Not sure what you played - check the pieces", "warn",
                     [inf.reason, "Centre the pieces on their squares; SPACE re-checks."])
            return
        if inf.kind == "invalid":
            self.highlight = inf.changed
            details = ["Changed squares: " + _names(inf.changed)]
            if inf.closest is not None and len(inf.closest_diff) <= 2:
                details.append(f"Closest legal move: {self.game.board.san(inf.closest)} "
                               f"(differs on {_names(inf.closest_diff)})")
                if self.game.board.is_capture(inf.closest):
                    details.append("Did you take the captured piece off the board?")
            if self.game.board.is_check():
                details.append("You are in CHECK.")
            details.append("Put the board right - it is re-checked automatically (H = hints).")
            self._ui("MOVE NOT ACCEPTED", "error", details)
            return
        san = self.game.board.san(inf.move)
        self.game.push(inf.move)
        self.last_move = inf.move
        self.highlight = []
        print(f"✓ You: {san}")
        self.details = [f"You played {san}"]
        if inf.move.promotion:
            self.details.append(f"Promotion: the pawn on {_printed_name(self.game, inf.move.to_square)} "
                                f"is now a {chess.piece_name(inf.move.promotion).upper()}")
        self._publish_state()
        if self.game.outcome() is not None:
            self._game_over()
        else:
            self._robot_turn()

    def _your_turn(self, details=None):
        check = " - you are in CHECK" if self.game.board.is_check() else ""
        self._ui(f"YOUR MOVE ({'white' if self.game.human_white else 'black'}){check}",
                 "your_turn", details)

    # ------------------------------------------------------------ robot
    def _robot_turn(self):
        self.state = State.ROBOT_THINKING
        self.pending_move = None
        self._ui("ROBOT THINKING ...", "robot", self.details)
        board = self.game.board.copy()

        def think():
            try:
                self.pending_move = self.engine.choose_move(board, self.level, self.ai_time_limit)
            except Exception as e:  # noqa: BLE001 - engine crashed: play something legal
                self.get_logger().error(f"engine: {e}")
                self.pending_move = next(iter(board.legal_moves))
        threading.Thread(target=think, daemon=True).start()

    def _send_robot_move(self, move):
        self.pending_move = None
        san = self.game.board.san(move)
        steps = self.game.plan_robot_move(move)
        geometry, travel = robot_geometry(self.game, steps, self.pieces, self.travel_margin,
                                          self.layout)
        msg = ChessRobotMove(uci=move.uci(), san=san, travel_height=float(travel))
        for g in geometry:
            msg.steps.append(PickPlace(pick_row=g.pick[0], pick_col=g.pick[1],
                                       place_row=g.place[0], place_col=g.place[1], piece=g.piece,
                                       grip_height=g.grip_height, grip_width=g.grip_width))
        self.game.push(move, steps)
        self.last_move = move
        self.move_pub.publish(msg)
        self._publish_state()
        print(f"Robot: {san}  ->  " + "; ".join(str(s) for s in steps)
              + f"  (travel {travel * 100:.1f} cm)")
        details = [d for d in self.details if d.startswith(("You played", "Promotion"))]
        details.append(f"Robot plays {san}")
        if move.promotion:
            details.append(f"Robot promotes: its pawn on {_printed_name(self.game, move.to_square)} "
                           f"is now a {chess.piece_name(move.promotion).upper()}")
        self.state = State.ROBOT_MOVING
        self._ui("ROBOT MOVING - keep your hands off the board", "robot", details)

    def robot_done_callback(self, msg):
        if not msg.robot_move_done or self.state != State.ROBOT_MOVING:
            return
        if msg.pick_failed:
            print("✗ The robot could not grip a piece - finish its move by hand")
            self._robot_finished()
            self._ui("Robot could not grip a piece - finish its move by hand", "warn",
                     ["Put the pieces where the game board shows them."])
            return
        print("✓ Robot finished")
        self._robot_finished()

    def _robot_finished(self):
        if self.game.outcome() is not None:
            self._game_over()
            return
        self.state = State.HUMAN_TURN
        self.verify_after_robot = True
        self.force_check = True
        self._your_turn(self.details)

    def _game_over(self):
        self.state = State.GAME_OVER
        out = self.game.outcome()
        if out.winner is None:
            text = f"GAME OVER - DRAW ({out.termination.name.lower().replace('_', ' ')})"
        else:
            human_won = out.winner == self.game.human_color
            text = f"GAME OVER - {'YOU WIN' if human_won else 'ROBOT WINS'} " \
                   f"({out.termination.name.lower()})"
        print("\n" + text + "\n" + " ".join(self.game.log))
        self._ui(text, "over", ["Set up the pieces and press S for a new game."])

    # ------------------------------------------------------------ messages
    def _publish_state(self):
        msg = ChessState(fen=self.game.board.fen() if self.game.started else "",
                         human_white=bool(self.game.human_white),
                         last_move_uci=self.last_move.uci() if self.last_move else "")
        if self.obs is not None and self.game.started:
            msg.mismatched_squares = [int(s) for s in self.game.mismatches(self.obs)]
        self.state_pub.publish(msg)

    # ------------------------------------------------------------ hands
    def _hands(self, work):
        vis = work.copy()
        cv2.polylines(vis, [self.model.corners.astype(np.int32)], True, (0, 255, 0), 2)
        if self.hands is None:
            return False, to_display(vis)
        small = cv2.resize(vis, None, fx=0.5, fy=0.5)
        res = self.hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        over = False
        if res.multi_hand_landmarks:
            h, w = vis.shape[:2]
            x0, y0 = self.model.corners.min(axis=0) - 40
            x1, y1 = self.model.corners.max(axis=0) + 40
            for lm in res.multi_hand_landmarks:
                xs = [pt.x * w for pt in lm.landmark]
                ys = [pt.y * h for pt in lm.landmark]
                hit = min(xs) < x1 and max(xs) > x0 and min(ys) < y1 and max(ys) > y0
                over = over or hit
                cv2.rectangle(vis, (int(min(xs)), int(min(ys))), (int(max(xs)), int(max(ys))),
                              (0, 0, 255) if hit else (255, 200, 0), 3)
        return over, to_display(vis)

    # ------------------------------------------------------------ drawing
    def _ui(self, text, kind="info", details=None):
        self.banner = (text, kind)
        if details is not None:
            self.details = list(details)

    def _detected_placement(self):
        if self.obs is None:
            return {}, []
        pred = self.obs.argmax(axis=1)
        conf = self.obs.max(axis=1)
        placement = {s: CLASSES[c] for s, c in enumerate(pred) if c}
        unsure = [s for s in range(64) if conf[s] < 0.6]
        return placement, unsure

    def _draw(self, work, cam_tile):
        detected, unsure = self._detected_placement()
        if self.game.started:
            b, hw = self.game.board, self.game.human_white
            placement = {logical_to_printed(sq, hw): p.symbol() for sq, p in b.piece_map().items()}
            promoted = [logical_to_printed(sq, hw) for sq in chess.SquareSet(b.promoted)]
            lm = self.last_move
            board_img = draw_diagram(placement, 704,
                                     [logical_to_printed(lm.from_square, hw)] if lm else [],
                                     [logical_to_printed(lm.to_square, hw)] if lm else [],
                                     marks=self.highlight, promoted=promoted)
        else:
            board_img = draw_diagram(detected, 704, marks=self.highlight,
                                     title="what the camera sees - press S to start")
        det_img = draw_diagram(detected, 288, marks=unsure, title="camera")
        now = time.time()
        settled = min(now - self.last_motion, now - self.last_hand + self.stable_time - 0.7)
        colour = "-" if not self.game.started else ("WHITE" if self.game.human_white else "BLACK")
        facts = [("you play", colour), ("engine", f"{self.engine.name} L{self.level}"),
                 ("promote to", chess.piece_name(self.game.promotion_piece)),
                 ("exposure", f"{self.camera.exposure_us / 1000:.0f} ms +{self.camera.gain_db:.0f} dB"),
                 ("unsure sq.", len(unsure)), ("motion", f"{self.motion:.1f} %"),
                 ("control", "LOCKED (R)" if self.control_locked else "UNLOCKED"),
                 ("state", self.state.name.lower().replace("_", " "))]
        status = {"banner": self.banner, "details": self.details, "facts": facts,
                  "stability": settled / max(self.stable_time, 1e-3), "log": self.game.move_pairs(),
                  "keys": "S start F force C colour 1-5 level P promo SPACE check R done L lock H Q"}
        if self.dashboard is not None:
            canvas = self.dashboard.render(board_img, cam_tile, det_img, status)
        else:
            canvas = np.hstack([board_img, cv2.resize(cam_tile, (588, 704))])
        cv2.imshow(WINDOW, canvas)
        self.window_open = True

    def _shutdown(self):
        self.capturing = False
        try:
            self.engine.close()
            self.camera.close()
            cv2.destroyAllWindows()
        finally:
            rclpy.try_shutdown()


def _names(printed_squares):
    return ", ".join(chess.square_name(s) for s in printed_squares) or "-"


def _printed_name(game, logical_sq):
    return chess.square_name(logical_to_printed(logical_sq, game.human_white))


def main(args=None):
    rclpy.init(args=args)
    node = ChessNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.capturing = False
        node.engine.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
