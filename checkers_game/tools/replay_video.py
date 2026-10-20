#!/usr/bin/env python3
"""Replay a recorded play video (capture_dataset.py -> video/frame_XXXX.jpg)
through the same detection + rules pipeline the node uses, and print the
moves it recognises.  Frame 0 must show the start position.

    python3 tools/replay_video.py ~/checkers_captures/<stamp>/video [--first white|black|auto]
                                  [--out /tmp/replay]   # annotated frames

In the recording a person may have moved both colours, so every move is
inferred for the side to move (no robot / AI involved).
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from checkers_game.camera import board_geometry as bg  # noqa: E402
from checkers_game.camera.board_detection import BoardDetection  # noqa: E402
from checkers_game.checkers.game import Game  # noqa: E402
from checkers_game.checkers.rules import BLACK, WHITE, color_name  # noqa: E402


def motion_of(prev, cur):
    if prev is None:
        return 0.0
    a = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(prev, (160, 160)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(cur, (160, 160)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
    return float((cv2.absdiff(a, b) > 25).mean() * 100)


def replay(frames, first="auto", stable_frames=3, motion_thr=1.5, out=None, verbose=True):
    img0 = cv2.imread(frames[0])
    corners, msg = bg.find_board_corners(img0)
    if corners is None:
        raise SystemExit(f"no board corners in frame 0: {msg}")
    det = BoardDetection(None, interactive=False)
    det.bounderies = corners
    ok, info = det.calibrate_from_start(img0)
    if not ok:
        raise SystemExit(f"frame 0 is not a recognisable start position: {info}")
    starts = [BLACK, WHITE] if first == "auto" else [BLACK if first == "black" else WHITE]
    results = []
    for first_color in starts:
        game = Game(human_first=(first_color == WHITE))
        game.start_standard(dark_parity=det.dark_parity)
        det.set_reference(img0, game.expected_grid(), learn=False)
        prev_w, last_grid, stable, last_sig = None, None, 0, None
        moves, rejected = [], 0
        for i, f in enumerate(frames):
            img = cv2.imread(f)
            res = det.analyse(img)
            grid = res["grid"]
            mot = motion_of(prev_w, res["warped"])
            prev_w = res["warped"]
            if last_grid is not None and np.array_equal(grid, last_grid) and mot < motion_thr:
                stable += 1
            else:
                stable = 0
            last_grid = grid
            if out:
                os.makedirs(out, exist_ok=True)
                cv2.imwrite(os.path.join(out, f"{i:04d}.jpg"),
                            det._overlay(res, f"frame {i} motion {mot:.1f} stable {stable}"))
            if stable < stable_frames:
                continue
            sig = grid.tobytes()
            if sig == last_sig:
                continue
            last_sig = sig
            mover = game.turn
            inf = game.infer_move(grid)
            if inf.kind == "no_change":
                continue
            if inf.accepted:
                game.apply_human(inf)
                det.set_reference(img, game.expected_grid())
                moves.append((i, color_name(mover), str(inf.move), inf.kind))
                if verbose:
                    print(f"  frame {i:4d}: {color_name(mover)} {inf.move}"
                          + ("  HUFF " + str(inf.huff_squares) if inf.kind == "huff" else "")
                          + (f"  (noise {inf.noise_squares})" if inf.noise_squares else ""))
            else:
                rejected += 1
                if verbose:
                    print(f"  frame {i:4d}: not accepted for {color_name(mover)}: {inf.reason}")
        results.append((len(moves), -rejected, first_color, moves, game))
        if verbose:
            print(f"  -> {len(moves)} moves recognised, {rejected} rejected "
                  f"(assuming {color_name(first_color)} moved first)\n")
    results.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return results[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--first", default="auto", choices=["auto", "white", "black"])
    ap.add_argument("--stable", type=int, default=3, help="frames the grid must be unchanged")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    frames = sorted(glob.glob(os.path.join(args.video_dir, "frame_*.jpg")))
    n, _, first, moves, game = replay(frames, args.first, args.stable, out=args.out)
    print(f"BEST: {n} moves with {color_name(first)} first")
    print(game.state.ascii())


if __name__ == "__main__":
    main()
