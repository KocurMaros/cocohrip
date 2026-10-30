#!/usr/bin/env python3
"""Evaluate a trained piece classifier (ONNX) on recorded sessions - the
way the game uses it.

    python3 eval_classifier.py ~/chess_models/latest/piece_classifier.onnx ~/chess_dataset/<s>

Reports per square / per piece / whole-board accuracy by lighting and
exposure, the confusion pairs, and - for photos taken after a game move -
whether ChessGame.infer_human_move recognises exactly that move from the
classifier's probabilities (the number that matters during play).
--show writes contact sheets of the wrong squares to <model dir>/errors/.
"""

import argparse
import glob
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from chess_game.dataset import placement_from_fen, read_labels  # noqa: E402
from chess_game.game import CLASSES, ChessGame  # noqa: E402
from chess_game.vision.board_model import BoardModel, contact_sheet, load_intrinsics  # noqa: E402
from chess_game.vision.classifier import (WORK_SCALE, SquareClassifier,  # noqa: E402
                                          square_crops)

CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chess_game", "vision",
                     "ximea_calibration_600.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--show", action="store_true", help="save crops of wrong squares")
    args = ap.parse_args()

    clf = SquareClassifier(os.path.expanduser(args.model))
    sessions = sorted({p for s in args.sessions for p in glob.glob(os.path.expanduser(s))})
    err_dir = os.path.join(os.path.dirname(os.path.expanduser(args.model)), "errors")
    stats = defaultdict(lambda: [0, 0])          # key -> [right, total]
    confusion = Counter()
    moves = defaultdict(lambda: [0, 0])
    intr = {}
    for d in sessions:
        for rec in read_labels(os.path.join(d, "labels.jsonl")):
            img = cv2.imread(os.path.join(d, rec["image"]), cv2.IMREAD_REDUCED_COLOR_2)
            if img is None:
                continue
            key = (img.shape[1], img.shape[0])
            if key not in intr:
                intr[key] = load_intrinsics(CALIB, key)
            model = BoardModel(np.asarray(rec["corners"]) * WORK_SCALE, *intr[key],
                               rec.get("a1_corner", "BR"))
            crops = square_crops(img, model)
            probs = clf.predict(crops)
            placement = placement_from_fen(rec["placement"])
            truth = np.array([CLASSES.index(placement[s]) if s in placement else 0 for s in range(64)])
            pred = probs.argmax(1)
            ok = pred == truth
            exp_tag = rec["image"].rsplit("_", 1)[-1].split(".")[0]          # e0 / e1 / e2
            for k in ("all", rec["lighting"], exp_tag):
                stats[("square", k)][0] += int(ok.sum())
                stats[("square", k)][1] += 64
                stats[("board", k)][0] += int(ok.all())
                stats[("board", k)][1] += 1
            for t, p in zip(truth[~ok], pred[~ok]):
                confusion[(CLASSES[t] if t else ".", CLASSES[p] if p else ".")] += 1
            if args.show and not ok.all():
                os.makedirs(err_dir, exist_ok=True)
                bad = [crops[s] if not ok[s] else np.zeros_like(crops[s]) for s in range(64)]
                labels = [f"{CLASSES[truth[s]]}>{CLASSES[pred[s]]}" if not ok[s] else "" for s in range(64)]
                cv2.imwrite(os.path.join(err_dir, f"{os.path.basename(d)}_{rec['image']}"),
                            contact_sheet(bad, labels))
            if rec.get("uci"):
                game = ChessGame()
                game.start(rec["fen_before"], human_white=rec["human_white"])
                inf = game.infer_move(probs)                    # either side's move
                hit = inf.kind == "move" and inf.move.uci() == rec["uci"]
                for k in ("all", rec["lighting"], exp_tag):
                    moves[k][0] += int(hit)
                    moves[k][1] += 1

    def pct(v):
        return f"{100.0 * v[0] / max(v[1], 1):6.2f}% ({v[1]})"
    print("square accuracy :", "  ".join(f"{k[1]} {pct(v)}" for k, v in sorted(stats.items())
                                         if k[0] == "square"))
    print("board accuracy  :", "  ".join(f"{k[1]} {pct(v)}" for k, v in sorted(stats.items())
                                         if k[0] == "board"))
    if moves:
        print("moves recognised:", "  ".join(f"{k} {pct(v)}" for k, v in sorted(moves.items())))
    print("most common confusions (truth > predicted):",
          ", ".join(f"{t}>{p} x{n}" for (t, p), n in confusion.most_common(12)))


if __name__ == "__main__":
    main()
