#!/usr/bin/env python3
"""How long does the human wait?  Replays recorded photos through the real
chess node code (camera, robot and ROS faked): position A still, a second
of "hand moving" frames, position B (A + one legal move) still - and
measures from B's first frame to the move being accepted.  Runs on any
machine with the dataset and a model, no robot needed.

    python3 latency_replay.py ~/chess_dataset/<session> ~/chess_models/latest/piece_classifier.onnx
    python3 latency_replay.py ... --backend cpu --stable-time 0.8 --moves 10
    REPLAY_DEBUG=1 python3 latency_replay.py ...      # node state after every move
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "test")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("model")
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "opencv"])
    ap.add_argument("--stable-time", type=float, default=None, help="default: the node's")
    ap.add_argument("--moves", type=int, default=8)
    ap.add_argument("--fps", type=float, default=20.0, help="camera frames per second")
    args = ap.parse_args()

    import pytest
    import cv2
    from test_node_sim import install_stubs
    mp = pytest.MonkeyPatch()
    install_stubs(mp)
    from chess_game import chess_node
    from chess_game.vision.classifier import SquareClassifier

    S = os.path.expanduser(args.session)
    recs = [json.loads(line) for line in open(os.path.join(S, "labels.jsonl"))]
    by_img = {r["image"]: r for r in recs}
    import chess
    from chess_game.dataset import placement_fen, placement_of
    pairs = []                                   # (photo before, photo after) of one game move
    for r in recs:
        if not (r["image"].endswith("_e0.jpg") and r.get("uci")) or r.get("exclude"):
            continue
        prev = by_img.get(f"{r['step'] - 1:05d}_e0.jpg")
        if prev is None or prev.get("exclude"):
            continue
        # both photos must show exactly the game's position (no human slips)
        board = chess.Board(r["fen_before"])
        hw = r["human_white"]
        if placement_fen(placement_of(board, hw)) != prev["placement"]:
            continue
        board.push_uci(r["uci"])
        if placement_fen(placement_of(board, hw)) == r["placement"]:
            pairs.append((prev, r))
    random.Random(1).shuffle(pairs)
    pairs = [p for p in pairs if p[1]["human_white"] == chess.Board(p[1]["fen_before"]).turn]
    print(f"{len(pairs)} recorded human moves usable")
    pairs = pairs[:args.moves]

    frames = {}
    current = {"img": None}

    class ReplayCamera:
        exposure_us, gain_db, serial = 40000, 0.0, "replay"

        def __init__(self):
            pass

        def grab(self, *a):
            time.sleep(1.0 / args.fps)           # like the real camera: wait for the next frame
            return current["img"]

        def auto_expose(self, measure, **kw):
            return self.exposure_us, 190.0

        def adjust(self, ratio):
            pass

        def close(self):
            pass

    first = cv2.imread(os.path.join(S, pairs[0][0]["image"]))
    current["img"] = first
    mp.setattr(chess_node, "ChessCamera", ReplayCamera)
    model_path = os.path.expanduser(args.model)
    mp.setattr(chess_node, "SquareClassifier",
               lambda path: SquareClassifier(model_path, prefer=args.backend))
    corners = np.asarray(pairs[0][0]["corners"], np.float32) * 0.5
    mp.setattr(chess_node.board_setup, "locate", lambda work, a1: (corners, "BR", "recorded"))
    for name in ("imshow", "namedWindow", "resizeWindow", "destroyAllWindows"):
        mp.setattr(cv2, name, lambda *a, **k: None)
    mp.setattr(cv2, "getWindowProperty", lambda *a: 1)
    mp.setattr(cv2, "waitKey", lambda *a: 255)
    sys.modules["test_node_sim"].PARAM_OVERRIDES["threaded_reading"] = True

    node = chess_node.ChessNode()
    if args.stable_time is not None:
        node.stable_time = args.stable_time
    node.engine = chess_node.create_engine("random")
    print(f"backend {node.classifier.backend}, stable_time {node.stable_time} s, "
          f"hand cooldown {node.hand_cooldown} s, {node.frames_per_read} readings")

    def run(seconds, img_fn, until=None):
        t_end = time.time() + seconds
        while time.time() < t_end:
            current["img"] = img_fn()
            with node.lock:
                node.latest = (current["img"], time.time())
            node.update()
            if until is not None and until():
                return True
            time.sleep(1.0 / args.fps)
        return False

    waits = []
    for before, after in pairs:
        a = frames.setdefault(before["image"], cv2.imread(os.path.join(S, before["image"])))
        b = frames.setdefault(after["image"], cv2.imread(os.path.join(S, after["image"])))
        hand = a.copy()                          # a skin-coloured arm reaching over the board
        h, w = a.shape[:2]
        rng = random.Random(after["step"])
        x, y = rng.randint(w // 4, 3 * w // 4), rng.randint(h // 4, 3 * h // 4)
        cv2.ellipse(hand, (x, y), (w // 9, h // 3), rng.uniform(0, 180), 0, 360, (120, 160, 210), -1)
        node.game.start(after["fen_before"], human_white=after["human_white"])
        if not node.game.human_to_move:          # a robot move in the recording: skip
            continue
        node.state = chess_node.State.HUMAN_TURN
        node.verify_after_robot = False
        node.pending_move = None
        node.highlight = []
        run(1.5, lambda: a)                      # board A, still
        flip = [False]

        def moving():
            flip[0] = not flip[0]
            return hand if flip[0] else a
        run(1.0, moving)                         # a hand moving over the board
        if not flip[0]:
            run(0.06, lambda: hand)              # ... and the hand is the last thing seen
        n_before = len(node.game.board.move_stack)
        t0 = time.time()
        ok = run(6.0, lambda: b, until=lambda: len(node.game.board.move_stack) > n_before)
        dt = time.time() - t0
        got = node.game.board.move_stack[n_before].uci() if ok else None
        waits.append(dt if ok and got == after["uci"] else None)
        if os.environ.get("REPLAY_DEBUG"):
            print("   banner:", node.banner, "| details:", node.details[:2], "| obs:", None if node.obs is None else node.obs.shape, "| settled:", node._settled(), "| reads:", len(node.reader._reads), "| obs_time<motion:", node.obs_time < node.last_motion)
        print(f"  {after['image'][:5]} {after['uci']}: "
              + (f"accepted after {dt:.2f} s" if ok else "not accepted within 6 s")
              + ("" if not ok or got == after["uci"] else f" (as {got}!)"))
    good = [w for w in waits if w is not None]
    if good:
        print(f"wait from hand off to move accepted: median {np.median(good):.2f} s, "
              f"max {max(good):.2f} s ({len(good)}/{len(waits)} moves right)", flush=True)
    node.reader.stop()
    time.sleep(0.5)
    sys.stdout.flush()
    os._exit(0)          # skip interpreter teardown under a running GPU session


if __name__ == "__main__":
    main()
