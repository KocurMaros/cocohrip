"""Simulated games through chess_node (no ROS, no camera, no robot):

* rclpy and the message packages are stubbed, OpenCV windows are no-ops;
* the "physical board" is a dict printed square -> piece symbol; the fake
  camera returns a frame that changes whenever it changes (motion), the
  fake classifier reads it with some uncertainty;
* a fake robot executes every ChessRobotMove (game-frame positions back to
  printed squares / storage) and answers on /robot_move_topic;
* a fake human plays random legal moves, taking captured pieces off and
  leaving promoted pawns as pawns.

    cd chess_game && python3 -m pytest test/test_node_sim.py -q
"""
import os
import random
import sys
import types

import chess
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_dataset import FIXTURE_CORNERS  # noqa: E402


# ---------------------------------------------------------------- stubs
class _Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class _Node:
    def __init__(self, name):
        self._params, self.pubs, self.subs = {}, {}, {}

    def declare_parameter(self, n, v):
        self._params[n] = v

    def get_parameter(self, n):
        return types.SimpleNamespace(value=self._params[n])

    def create_publisher(self, t, topic, q):
        self.pubs[topic] = _Pub()
        return self.pubs[topic]

    def create_subscription(self, t, topic, cb, q):
        self.subs[topic] = cb

    def create_timer(self, period, cb):
        return None

    def get_logger(self):
        return types.SimpleNamespace(info=print, warn=print, warning=print, error=print)

    def destroy_node(self):
        pass


def _msg(name, **defaults):
    def init(self, **kw):
        for k, v in defaults.items():
            setattr(self, k, list(v) if isinstance(v, list) else v)
        for k, v in kw.items():
            setattr(self, k, v)
    return type(name, (), {"__init__": init})


def install_stubs(monkeypatch):
    rclpy = types.ModuleType("rclpy")
    rclpy.try_shutdown = lambda: None
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _Node
    rclpy.node = node_mod
    cm = types.ModuleType("checkers_msgs.msg")
    cm.RobotMove = _msg("RobotMove", robot_move_done=False, pick_failed=False, pick_row=0, pick_col=0)
    chm = types.ModuleType("chess_msgs.msg")
    chm.ChessRobotMove = _msg("ChessRobotMove", uci="", san="", steps=[], travel_height=0.0)
    chm.PickPlace = _msg("PickPlace", pick_row=0.0, pick_col=0.0, place_row=0.0, place_col=0.0,
                         piece="", grip_height=0.0, grip_width=0.0, offset_row=0.0, offset_col=0.0)
    chm.ChessState = _msg("ChessState", fen="", human_white=True, last_move_uci="",
                          mismatched_squares=[])
    for name, mod in {"rclpy": rclpy, "rclpy.node": node_mod,
                      "checkers_msgs": types.ModuleType("checkers_msgs"), "checkers_msgs.msg": cm,
                      "chess_msgs": types.ModuleType("chess_msgs"), "chess_msgs.msg": chm,
                      "mediapipe": None}.items():
        monkeypatch.setitem(sys.modules, name, mod)


class World:
    """The physical board + fake camera / classifier / robot / human."""

    def __init__(self, seed, human_white=True):
        from chess_game.dataset import placement_of
        self.rng = random.Random(seed)
        self.placement = placement_of(chess.Board(), human_white)
        self.storage = {}
        self.version = 0

    # camera: a frame that differs whenever the board changed
    def frame(self):
        img = np.full((2056, 2464, 3), 120, np.uint8)
        r = random.Random(self.version)
        for _ in range(30):
            x, y = r.randrange(0, 2300), r.randrange(0, 1900)
            img[y:y + 150, x:x + 150] = r.randrange(0, 255)
        return img

    def changed(self):
        self.version += 1

    def obs(self):
        from chess_game.game import CLASSES
        o = np.full((64, 13), 0.02 / 12)
        for s in range(64):
            c = CLASSES.index(self.placement[s]) if s in self.placement else 0
            o[s, c] = 0.98
        return o


def run_game(monkeypatch, seed, human_white, engine="random", max_plies=80, robot_fails=False):
    install_stubs(monkeypatch)
    import cv2
    for mod in [m for m in sys.modules if m.startswith("chess_game.chess_node")]:
        del sys.modules[mod]
    from chess_game import chess_node
    from chess_game.frames import game_to_printed, printed_to_logical

    world = World(seed, human_white)
    keys = []
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a: keys.pop(0) if keys else 255)

    class FakeCamera:
        exposure_us = 40000
        gain_db = 0.0

        def __init__(self):
            pass

        def grab(self, *a):
            return world.frame()

        def auto_expose(self, measure, **kw):
            return self.exposure_us, 190.0

        def set_exposure(self, us):
            return self.exposure_us

        def adjust(self, ratio):
            pass

        def close(self):
            pass

    class FakeClassifier:
        def __init__(self, path):
            pass

        def classify(self, image, model):
            return world.obs()

    monkeypatch.setattr(chess_node, "ChessCamera", FakeCamera)
    monkeypatch.setattr(chess_node, "SquareClassifier", FakeClassifier)
    monkeypatch.setattr(chess_node.board_setup, "locate",
                        lambda work, a1: (np.float32(FIXTURE_CORNERS), "BR", "fixture"))
    node = chess_node.ChessNode()
    node._params = None
    node.stable_time = 0.0
    node.engine = chess_node.create_engine(engine)
    node.engine.rng = random.Random(seed) if hasattr(node.engine, "rng") else None

    def tick(n=3):
        for _ in range(n):
            with node.lock:
                node.latest = world.frame()
            node.update()

    def game_to_place(row, col):
        if 0 <= row <= 7 and 0 <= col <= 7 and float(row).is_integer() and float(col).is_integer():
            return ("sq", game_to_printed(int(row), int(col)))
        return ("store", (row, col))

    def robot_execute(msg):
        for st in msg.steps:
            src = game_to_place(st.pick_row, st.pick_col)
            dst = game_to_place(st.place_row, st.place_col)
            assert src[0] == "sq", "robot picks only from the board"
            piece = world.placement.pop(src[1])
            assert piece == st.piece, f"robot expected {st.piece} on {src}, found {piece}"
            assert st.grip_height > 0 and st.grip_width > 0
            if dst[0] == "sq":
                assert dst[1] not in world.placement, "robot places onto an occupied square"
                world.placement[dst[1]] = piece
            else:
                assert dst[1] not in world.storage, "storage slot used twice"
                world.storage[dst[1]] = piece
        assert msg.travel_height > 0.05
        world.changed()

    def human_move():
        board = node.game.board
        move = world.rng.choice(list(board.legal_moves))
        if move.promotion:      # the human promotes to whatever the node expects
            move = chess.Move(move.from_square, move.to_square, node.game.promotion_piece)
        hw = node.game.human_white
        p = (lambda s: s if hw else 63 - s)                      # logical -> printed
        if board.is_en_passant(move):
            world.placement.pop(p(chess.square(chess.square_file(move.to_square),
                                               chess.square_rank(move.from_square))))
        if board.is_castling(move):
            rank = chess.square_rank(move.from_square)
            ks = chess.square_file(move.to_square) > chess.square_file(move.from_square)
            rf, rt = chess.square(7 if ks else 0, rank), chess.square(5 if ks else 3, rank)
            world.placement[p(rt)] = world.placement.pop(p(rf))
        piece = world.placement.pop(p(move.from_square))       # a promoted pawn stays a pawn
        world.placement[p(move.to_square)] = piece
        world.changed()
        return move

    tick()
    keys.append(ord('s'))
    tick()
    assert node.game.started and node.game.human_white == human_white
    plies = 0
    robot_pub = node.pubs["/chess/robot_move"]
    while plies < max_plies and node.state != chess_node.State.GAME_OVER:
        if node.state == chess_node.State.HUMAN_TURN:
            before = len(node.game.board.move_stack)
            move = human_move()
            tick()
            # (the robot may already have answered within these ticks)
            assert len(node.game.board.move_stack) > before, node.banner
            assert node.game.board.move_stack[before] == move
        elif node.state == chess_node.State.ROBOT_THINKING:
            tick()
        elif node.state == chess_node.State.ROBOT_MOVING:
            robot_execute(robot_pub.msgs[-1])
            node.subs["/robot_move_topic"](sys.modules["checkers_msgs.msg"].RobotMove(
                robot_move_done=True))
            tick()
        plies += 1
        if node.state == chess_node.State.HUMAN_TURN:
            # the physical board shows the game position (as it looks)
            assert node.game.mismatches(world.obs()) == []
        tick(1)
    if node.state == chess_node.State.ROBOT_MOVING:
        robot_execute(robot_pub.msgs[-1])
    return node, world, printed_to_logical


@pytest.mark.parametrize("seed,human_white", [(1, True), (2, False), (3, True), (4, False)])
def test_random_games_stay_in_sync(monkeypatch, seed, human_white):
    node, world, _ = run_game(monkeypatch, seed, human_white)
    assert len(node.game.board.move_stack) > 10
    # every piece the robot captured is in its own storage slot
    assert len(world.storage) == node.game.storage_used


def test_robot_first_when_human_plays_black(monkeypatch):
    node, world, _ = run_game(monkeypatch, 7, False, max_plies=2)
    assert node.game.log[0].startswith("1. ")      # white = robot moved first
