#!/usr/bin/env python3
"""Evaluate board + piece detection on a dataset recorded with
capture_dataset.py.

    python3 tools/eval_detection.py ~/checkers_captures/20261015_144204 [--show]

For every (train scene, test scene) pair of the same camera profile it
calibrates on the train scene's start position and classifies the test
scene's dark squares, reporting per-square accuracy.  Ground truth: start
and empty scenes are known; other arrangements can be labelled in
labels.json next to meta.json:  {"midgame": {"white": [[r,c],...],
"black": [[r,c],...]}}  (game orientation: robot's black pieces on top).
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from checkers_game.camera import board_geometry as bg  # noqa: E402
from checkers_game.camera import board_vision as bv  # noqa: E402
from checkers_game.camera.board_detection import BoardDetection  # noqa: E402
from checkers_game.checkers.rules import GameState  # noqa: E402

CLASSES = bv.PieceClassifier.CLASSES


def truth_grid(arrangement, labels):
    if arrangement in ("start", "hand"):
        return np.array(GameState.initial(1).to_grid())
    if arrangement == "empty":
        return np.zeros((8, 8), int)
    if arrangement in labels:
        g = np.zeros((8, 8), int)
        for r, c in labels[arrangement].get("white", []):
            g[r, c] = 1
        for r, c in labels[arrangement].get("black", []):
            g[r, c] = 2
        return g
    return None


def load(dataset):
    frames = {}
    for f in sorted(glob.glob(os.path.join(dataset, "*", "*_[0-9].png"))):
        scene = os.path.basename(os.path.dirname(f))
        prof = os.path.basename(f).rsplit("_", 1)[0]
        frames.setdefault((scene, prof), []).append(f)
    lab_path = os.path.join(dataset, "labels.json")
    labels = json.load(open(lab_path)) if os.path.exists(lab_path) else {}
    return frames, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--profiles", default="current,exp40k,exp20k")
    ap.add_argument("--show", action="store_true", help="write overlay images to <dataset>/eval")
    ap.add_argument("--no-calib", action="store_true",
                    help="use the default (uncalibrated) piece model")
    args = ap.parse_args()
    frames, labels = load(args.dataset)
    profiles = args.profiles.split(",")
    out_dir = os.path.join(args.dataset, "eval")
    if args.show:
        os.makedirs(out_dir, exist_ok=True)

    total = {}
    for prof in profiles:
        train_scenes = [s for (s, p) in frames if p == prof and s.startswith("start__")]
        for train in train_scenes:
            img = cv2.imread(frames[(train, prof)][0])
            corners, msg = bg.find_board_corners(img)
            if corners is None:
                print(f"{prof:8s} {train}: no corners ({msg})")
                continue
            det = BoardDetection(None, interactive=False)
            det.bounderies = corners
            ok, info = det.calibrate_from_start(img)
            if not ok:
                print(f"{prof:8s} {train}: calibration failed ({info})")
                continue
            sep = det._class_separation(*_feats_labels(det, img))
            if args.no_calib:
                det.classifier = bv.PieceClassifier()
            print(f"\n== profile {prof}, calibrated on {train} (rot {det.rotation}, "
                  f"gap piece/empty {sep[0]:.2f} white/black {sep[1]:.2f})")
            for (scene, p), files in sorted(frames.items()):
                if p != prof:
                    continue
                arrangement = scene.split("__")[0]
                gt = truth_grid(arrangement, labels)
                if gt is None or arrangement == "hand":
                    continue
                for f in files:
                    im = cv2.imread(f)
                    res = det.analyse(im)
                    grid = res["grid"]
                    errs = []
                    for r in range(8):
                        for c in range(8):
                            if bv.is_dark(r, c, det.dark_parity) and grid[r, c] != gt[r, c]:
                                errs.append((r, c, int(gt[r, c]), int(grid[r, c])))
                    key = (prof, scene.split("__")[1] == train.split("__")[1])
                    t = total.setdefault(key, [0, 0])
                    t[0] += 32 - len(errs)
                    t[1] += 32
                    tag = "same-light" if key[1] else "OTHER-light"
                    print(f"   {os.path.relpath(f, args.dataset):38s} {tag:11s} "
                          f"errors {len(errs):2d}  {errs[:6]}")
                    if args.show:
                        vis = det._overlay(res, f"{train} -> {os.path.relpath(f, args.dataset)}")
                        name = f"{prof}__{train}__{scene}__{os.path.basename(f)}"
                        cv2.imwrite(os.path.join(out_dir, name.replace('.png', '.jpg')), vis)
    print("\nSUMMARY (dark-square accuracy, absolute classification, no temporal info)")
    for (prof, same), (ok, n) in sorted(total.items()):
        print(f"  {prof:8s} {'same lighting ' if same else 'other lighting'}: {ok}/{n} = {ok / n:.3f}")


def _feats_labels(det, img):
    w = det.warp(img)
    g = bv.to_gray(w)
    norm, _, _ = bv.normalise(g, det.dark_parity)
    cells, feats = bv.cell_features(w, norm, det.dark_parity)
    _, labels = bv.start_position_labels(det.dark_parity)
    return feats, labels


if __name__ == "__main__":
    main()
