"""Simulated end-to-end run of checkers_node (no ROS, no camera, no robot):

* rclpy / checkers_msgs / ximea are stubbed, OpenCV windows are no-ops,
  pygame uses the dummy video driver;
* a fake camera renders the *physical* board (synthetic_board) every frame;
* a fake robot executes every /move_topic message on the physical board and
  answers on /robot_move_topic;
* a fake human plays random moves - sometimes ignoring a mandatory capture
  so the huff path is exercised.

    cd checkers_game && python3 -m pytest test/test_node_sim.py -q
"""
import os
import random
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# ---------------------------------------------------------------- stubs
class _Pub:
    def __init__(self, topic):
        self.topic = topic
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class _Logger:
    def info(self, m):
        pass

    warn = warning = error = info


class _Node:
    def __init__(self, name):
        self._params, self.pubs, self.subs = {}, {}, {}

    def declare_parameter(self, n, v):
        self._params[n] = v

    def get_parameter(self, n):
        return types.SimpleNamespace(value=self._params[n])

    def create_publisher(self, t, topic, q):
        self.pubs[topic] = _Pub(topic)
        return self.pubs[topic]

    def create_subscription(self, t, topic, cb, q):
        self.subs[topic] = cb

    def create_timer(self, period, cb):
        return None

    def get_logger(self):
        return _Logger()

    def destroy_node(self):
        pass


def _msg_class(name, **defaults):
    def init(self, **kw):
        for k, v in defaults.items():
            setattr(self, k, list(v) if isinstance(v, list) else v)
        for k, v in kw.items():
            setattr(self, k, v)
    return type(name, (), {"__init__": init})


def install_stubs(monkeypatch):
    rclpy = types.ModuleType("rclpy")
    rclpy.try_shutdown = lambda: None
    rclpy.init = lambda args=None: None
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _Node
    rclpy.node = node_mod
    msgs = types.ModuleType("checkers_msgs.msg")
    msgs.Piece = _msg_class("Piece", row=0, col=0, color="", king=False)
    msgs.Board = _msg_class("Board", pieces=[])
    msgs.Move = _msg_class("Move", target_row=0, target_col=0, piece_for_moving=None,
                           removed_pieces=[])
    msgs.HandDetected = _msg_class("HandDetected", hand_detected=False)
    msgs.RobotMove = _msg_class("RobotMove", robot_move_done=False)
    pkg = types.ModuleType("checkers_msgs")
    pkg.msg = msgs
    ximea = types.ModuleType("ximea")
    ximea.xiapi = None
    for name, mod in [("rclpy", rclpy), ("rclpy.node", node_mod), ("checkers_msgs", pkg),
                      ("checkers_msgs.msg", msgs), ("ximea", ximea), ("mediapipe", None)]:
        monkeypatch.setitem(sys.modules, name, mod)
    import cv2
    keys = []
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "createTrackbar", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "getTrackbarPos", lambda *a, **k: 2)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a, **k: 1.0)
    monkeypatch.setattr(cv2, "waitKey", lambda *a, **k: keys.pop(0) if keys else 255)
    return keys


# ------------------------------------------------------------ fake world
class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    @staticmethod
    def sleep(s):
        import time as _t
        _t.sleep(min(s, 0.001))


class FakeCamera:
    """Renders the physical board; lighting can be changed on the fly."""

    def __init__(self, world):
        self.world = world
        self.exposure_us = 40000

    def get_camera_image(self):
        import synthetic_board as sb
        w = self.world
        w.frame += 1
        # piece placement only changes when the board changes; sensor noise
        # changes every frame
        td = sb.render_topdown(w.grid, rng=np.random.RandomState(int(w.grid.sum()) % 97))
        return sb.to_camera(td, w.corners, rotate_k=1, gain=w.gain,
                            rng=np.random.RandomState(w.frame))

    def auto_expose(self, *a, **k):
        return self.exposure_us

    def set_exposure(self, e):
        self.exposure_us = e

    def track_exposure(self, *a, **k):
        return False


class World:
    def __init__(self, seed=0):
        import synthetic_board as sb
        from checkers_game.checkers.rules import GameState
        self.grid = np.array(GameState.initial(1).to_grid())
        self.corners = sb.random_corners(np.random.RandomState(seed))
        self.frame = 0
        self.gain = 1.0


def run_until(node, clock, cond, max_steps=200):
    for _ in range(max_steps):
        clock.t += 0.1
        node.update()
        if cond():
            return True
    return False


@pytest.mark.parametrize("seed", [0, 1])
def test_node_plays_full_game(monkeypatch, tmp_path, seed):
    monkeypatch.setenv("HOME", str(tmp_path))        # no cached calibration
    keys = install_stubs(monkeypatch)
    import importlib
    import checkers_game.camera.board_detection as bd
    monkeypatch.setattr(bd, "CALIBRATION_DIR", str(tmp_path / ".cocohrip"))
    monkeypatch.setattr(bd, "CALIBRATION_PATH", str(tmp_path / ".cocohrip" / "c.json"))
    node_mod = importlib.import_module("checkers_game.checkers_node")
    node_mod = importlib.reload(node_mod)

    world = World(seed)
    clock = FakeClock()
    monkeypatch.setattr(node_mod, "time", clock)
    monkeypatch.setattr(node_mod, "XimeaCamera", lambda: FakeCamera(world))

    keys += [32, 32, 32]                    # camera ok, corners ok, piece calibration ok
    node = node_mod.CheckersNode()
    node.capture_running = False            # frames are pulled directly (_grab fallback)
    with node.cam_lock:
        node.latest_camera_image = None
    node._grab = lambda: node.ximeaCamera.get_camera_image()

    from checkers_game.checkers.game import HUMAN, ROBOT
    rng = random.Random(seed)
    move_pub = node.pubs["/move_topic"]
    S = node_mod.GameState

    keys.append(ord('s'))
    assert run_until(node, clock, lambda: node.game.started)
    huffs = 0
    for ply in range(60):
        if node.game_state == S.GAME_OVER:
            break
        if node.game_state == S.ROBOT_MOVING:
            msg = move_pub.msgs[-1]
            for p in msg.removed_pieces:
                assert world.grid[p.row, p.col] != 0
                world.grid[p.row, p.col] = 0
            s = msg.piece_for_moving
            assert world.grid[s.row, s.col] == 2
            world.grid[s.row, s.col] = 0
            world.grid[msg.target_row, msg.target_col] = 2
            node.robot_move_callback(sys.modules["checkers_msgs.msg"].RobotMove(robot_move_done=True))
            assert np.array_equal(world.grid, np.array(node.game.expected_grid()))
            continue
        assert node.game_state == S.PLAYER_TURN and node.game.turn == HUMAN
        st = node.game.state
        huffable = st.huffable_moves()
        if huffable and rng.random() < 0.5:
            m = rng.choice(huffable)
            huffs += 1
        else:
            m = rng.choice(st.legal_moves())
        after = st.apply(m).to_grid()
        world.grid = np.array(after)
        n_before = len(node.game.log)
        assert run_until(node, clock, lambda: len(node.game.log) > n_before
                         or node.game_state == S.GAME_OVER), \
            f"node did not accept {m}"
        assert node.game.log[n_before].startswith("WHITE")
    assert node.game.state is not None
    # sanity: the robot's published moves were all legal at the time (checked
    # implicitly by the physical-board asserts above)
    assert len(move_pub.msgs) > 3
