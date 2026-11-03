#!/usr/bin/env python3
"""Check (and repair) the labels of a recorded session against the photos.

Between two consecutive photos exactly the squares the instruction names
should change.  `check` lists every photo whose label disagrees with what
the camera saw (moved squares that look unchanged, or changes elsewhere).
`rebuild` re-creates the real board of each photo from the instructions the
camera saw being carried out (all their squares changed); instructions that
were passed over are left out.  Use it when photos were taken before the
move was made, or a move was skipped.  It needs a photo up to which the
labels are known to be right.

    python3 check_dataset.py ~/chess_dataset/<session>
    python3 check_dataset.py ~/chess_dataset/<session> --rebuild-from 6          # dry run
    python3 check_dataset.py ~/chess_dataset/<session> --rebuild-from 6 --apply  # rewrite labels
    python3 check_dataset.py ~/chess_dataset/<session> --model m.onnx [--fix]   # piece types

--model: a trained classifier looks at every photo.  Squares where it is
sure (all three exposures agree, p > 0.9, label p < 0.05) that the photo
shows another piece than the label are listed - typically a piece placed
with the wrong type or colour, which the change detection cannot see.
--fix relabels exactly those squares (old label kept in "model_fixed").

--apply keeps the old file as labels.before_rebuild.jsonl and the old label
of every changed photo in its "relabelled_from" field.
"""

import argparse
import json
import os
import shutil
import sys

import chess
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from chess_game.dataset import (ChangeDetector, placement_fen,  # noqa: E402
                                placement_from_fen, read_labels)
from chess_game.vision.board_model import BoardModel, load_intrinsics  # noqa: E402

CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chess_game", "vision",
                     "ximea_calibration_600.npz")
CHANGED = 8.0          # change score that counts as "this square changed"


def diff(a, b):
    return {q for q in set(a) | set(b) if a.get(q) != b.get(q)}


def names(squares):
    return ",".join(chess.square_name(q) for q in sorted(squares)) or "-"


def check_with_model(S, model_path, fix):
    from chess_game.game import CLASSES
    from chess_game.vision.classifier import SquareClassifier, square_crops
    clf = SquareClassifier(model_path)
    recs = read_labels(os.path.join(S, "labels.jsonl"))
    by_step = {}
    for r in recs:
        by_step.setdefault(r["step"], []).append(r)
    intr = None
    fixes = {}
    for step, group in sorted(by_step.items()):
        placement = placement_from_fen(group[0]["placement"])
        labels = np.array([CLASSES.index(placement[q]) if q in placement else 0 for q in range(64)])
        seen_all = []
        for r in group:
            img = cv2.imread(os.path.join(S, r["image"]), cv2.IMREAD_REDUCED_COLOR_2)
            if intr is None:
                intr = load_intrinsics(CALIB, (img.shape[1], img.shape[0]))
            model = BoardModel(np.asarray(r["corners"]) / 2.0, *intr, r.get("a1_corner", "BR"))
            seen_all.append(clf.predict(square_crops(img, model)))
        sure = {}
        for q in range(64):
            tops = {int(np.argmax(p[q])) for p in seen_all}
            if len(tops) == 1:
                top = tops.pop()
                if top != labels[q] and all(p[q, top] > 0.9 and p[q, labels[q]] < 0.05
                                            for p in seen_all):
                    sure[q] = top
        if sure:
            fixes[step] = sure
            print(f"{step:05d} " + "; ".join(
                f"{chess.square_name(q)}: label {CLASSES[labels[q]]} -> looks like {CLASSES[c]}"
                for q, c in sorted(sure.items())))
    print(f"{len(by_step)} positions, {len(fixes)} with confident disagreements")
    if fix and fixes:
        shutil.copy(os.path.join(S, "labels.jsonl"), os.path.join(S, "labels.before_model_fix.jsonl"))
        with open(os.path.join(S, "labels.jsonl"), "w") as f:
            for r in recs:
                if r["step"] in fixes:
                    placement = placement_from_fen(r["placement"])
                    for q, c in fixes[r["step"]].items():
                        if c:
                            placement[q] = CLASSES[c]
                        else:
                            placement.pop(q, None)
                    r.setdefault("model_fixed", r["placement"])
                    r["placement"] = placement_fen(placement)
                f.write(json.dumps(r) + "\n")
        print("labels.jsonl rewritten (old: labels.before_model_fix.jsonl)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--rebuild-from", type=int, help="last photo whose label is known to be right")
    ap.add_argument("--apply", action="store_true", help="rewrite labels.jsonl (with --rebuild-from)")
    ap.add_argument("--model", help="piece classifier (.onnx) to check the piece types")
    ap.add_argument("--fix", action="store_true", help="relabel what --model is sure about")
    args = ap.parse_args()
    if args.model:
        return check_with_model(os.path.expanduser(args.session), os.path.expanduser(args.model),
                                args.fix)
    S = os.path.expanduser(args.session)
    recs = read_labels(os.path.join(S, "labels.jsonl"))
    photos = [r for r in recs if r["image"].endswith("_e0.jpg")]
    labels = [placement_from_fen(r["placement"]) for r in photos]
    intr = None
    scores = [None]
    for a, b in zip(photos, photos[1:]):
        ia = cv2.imread(os.path.join(S, a["image"]), cv2.IMREAD_REDUCED_COLOR_2)
        ib = cv2.imread(os.path.join(S, b["image"]), cv2.IMREAD_REDUCED_COLOR_2)
        if intr is None:
            intr = load_intrinsics(CALIB, (ib.shape[1], ib.shape[0]))
        model = BoardModel(np.asarray(b["corners"]) / 2.0, *intr, b.get("a1_corner", "BR"))
        det = ChangeDetector(model, ib.shape)
        scores.append((det.scores(ia, ib), det))

    if args.rebuild_from is None:
        bad = 0
        for i in range(1, len(photos)):
            if photos[i]["kind"] == "reset":
                continue
            s, det = scores[i]
            moved = diff(labels[i - 1], labels[i])
            seen = {q for q in range(64) if s[q] > CHANGED}
            missing = {q for q in moved if s[q] <= CHANGED}
            extra = seen - moved - det.allowed(moved)
            if missing or extra:
                bad += 1
                print(f"{photos[i]['image'][:5]} SUSPECT  label moves {names(moved)}, camera saw "
                      f"{names(seen)}  (missing {names(missing)}, extra {names(extra)}, "
                      f"trigger {photos[i].get('trigger', '?')})")
        print(f"{len(photos)} photos, {bad} suspect")
        return

    start = args.rebuild_from
    board, done = dict(labels[start]), start
    truth = {p["step"]: p["placement"] for p in photos[:start + 1]}
    for i in range(start + 1, len(photos)):
        if photos[i]["kind"] == "reset":           # a reset is confirmed by the human (SPACE)
            board, done = dict(labels[i]), i
        else:
            s, _ = scores[i]
            seen = {q for q in range(64) if s[q] > CHANGED}
            for k in range(done + 1, i + 1):
                sq = diff(labels[k - 1], labels[k])
                if sq and sq <= seen:
                    for q in sq:
                        if q in labels[k]:
                            board[q] = labels[k][q]
                        else:
                            board.pop(q, None)
                    done = k
        truth[photos[i]["step"]] = placement_fen(board)
        state = "ok" if truth[photos[i]["step"]] == photos[i]["placement"] else "RELABEL"
        print(f"{photos[i]['image'][:5]} {state}")
    print("last photo shows:\n" + chess.Board(truth[photos[-1]["step"]] + " w - - 0 1")
          .unicode(empty_square="."))
    if args.apply:
        shutil.copy(os.path.join(S, "labels.jsonl"), os.path.join(S, "labels.before_rebuild.jsonl"))
        with open(os.path.join(S, "labels.jsonl"), "w") as f:
            for r in recs:
                if r["placement"] != truth[r["step"]]:
                    r.setdefault("relabelled_from", r["placement"])
                    r["placement"] = truth[r["step"]]
                f.write(json.dumps(r) + "\n")
        print("labels.jsonl rewritten (old: labels.before_rebuild.jsonl)")


if __name__ == "__main__":
    main()
