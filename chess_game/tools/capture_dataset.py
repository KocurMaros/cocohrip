#!/usr/bin/env python3
"""Record a labelled photo dataset of the chess pieces - guided.

Run on the robot PC with its display (stop the game nodes first, they hold
the camera):

    source ~/checkers_ws/install/setup.bash
    python3 ~/chess_ws/src/cocohrip/chess_game/tools/capture_dataset.py --lighting normal
    python3 .../capture_dataset.py --resume ~/chess_dataset/<session>
    python3 .../capture_dataset.py --replay a.jpg b.jpg        # no camera

The window shows the camera (the board as you see it from your side), the
position to set up next - red frame = take from, green frame = put on -
and one line saying what to do.  Do it and take your hand away.  When the
board is still and changed exactly as asked, three photos (three exposures)
are taken automatically and the next instruction appears.  Resetting the
board (empty / start position) always waits for SPACE.

Keys:  SPACE  take the photos now - only needed for resets, or when the
              automatic photo does not come (it needs the board to have moved)
       N      skip this instruction (e.g. a piece is missing)
       U      the last photos were wrong: delete them, fix the board, SPACE
       1 2 3  lighting tag: 1 normal, 2 dark, 3 bright (set it whenever you
              change the lights - it only labels the photos)
       E      auto-exposure now        B  re-detect the board corners
       + / -  change-detection threshold        Q / ESC  quit (resume later)

Output: <out>/<session>/NNNNN_e<k>.jpg (full resolution, raw orientation)
and labels.jsonl - one JSON line per photo: printed piece placement (FEN),
board corners, exposure, lighting tag (chess_game.dataset.label_record).
"""

import argparse
import glob
import json
import os
import sys
import time

import chess
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from chess_game.dataset import (ChangeDetector, DatasetPlanner, Instruction,  # noqa: E402
                                check_change, label_record, placement_from_fen, read_labels)
from chess_game.ui.diagram import draw_diagram  # noqa: E402
from chess_game.vision.board_model import BoardModel, load_intrinsics  # noqa: E402
from chess_game.vision.board_setup import click_corners, find_board_corners, locate  # noqa: E402
from chess_game.vision.camera import ImageFolderCamera, to_display, white_level  # noqa: E402
from chess_game.game import CLASSES  # noqa: E402
from chess_game.vision.classifier import SquareClassifier, square_crops  # noqa: E402

WIN = "Chess dataset"
CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chess_game", "vision",
                     "ximea_calibration_600.npz")
WORK_SCALE = 0.5                       # detection / change checks on half resolution
BRACKET = (1.0, 0.55, 1.8)             # exposures per position, x the base exposure
LIGHTINGS = ["normal", "dark", "bright"]       # dark room / normal / too much light
STILL_S = 1.0                          # board must be still this long before auto photos
LIGHT_CHANGE_EVERY = 150               # remind to change the lights every N positions


def work_image(frame):
    return cv2.resize(frame, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)


class Recorder:
    def __init__(self, args):
        self.args = args
        if args.replay:
            self.cam = ImageFolderCamera(args.replay)
        else:
            from chess_game.vision.camera import ChessCamera
            self.cam = ChessCamera()
        frame = self.cam.grab()
        self.full_size = (frame.shape[1], frame.shape[0])
        self.work_shape = work_image(frame).shape
        self.K, self.dist = load_intrinsics(CALIB, (self.work_shape[1], self.work_shape[0]))
        self.a1_corner = args.a1_corner

        if args.resume:
            self.dir = os.path.expanduser(args.resume)
            labels = read_labels(os.path.join(self.dir, "labels.jsonl"))
            placement = placement_from_fen(labels[-1]["placement"]) if labels else {}
            self.photos = len({r["step"] for r in labels})
            self.lighting = args.lighting or (labels[-1]["lighting"] if labels else "normal")
        else:
            self.dir = os.path.join(os.path.expanduser(args.out), time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(self.dir)
            labels, placement, self.photos, self.lighting = [], {}, 0, args.lighting or "normal"
        self.labels_path = os.path.join(self.dir, "labels.jsonl")
        with open(os.path.join(self.dir, "session.json"), "w") as f:
            json.dump({"camera": self.cam.serial, "full_size": self.full_size,
                       "calibration": "ximea_calibration_600.npz", "bracket": BRACKET}, f)

        self.planner = DatasetPlanner(placement, seed=args.seed or int(time.time()))
        for r in labels:
            if r["image"].endswith("_e0.jpg"):
                self.planner.record(placement_from_fen(r["placement"]))
        if args.resume and placement:
            self.planner.skip()            # continue scattering from the stored position
        elif args.resume:
            self.planner.after_empty_board()   # empty board recorded: set up the start next
        self.ref = None                    # work image of the last photo (change reference)
        self.retake = None                 # Instruction to re-photograph after U
        if args.resume and args.start:     # restarted during a reset: ask for it again
            self.planner.force_start(args.start == "white")
        elif args.resume:                  # first make sure the board is where we left off
            self.retake = Instruction("Resumed: make the board look exactly like the diagram, "
                                      "then SPACE", dict(placement), dict(placement), kind="reset")
        self.threshold = args.threshold
        # a trained piece classifier checks every photo against its label
        # (catches a queen placed instead of a rook, a black pawn for a white one)
        self.checker = SquareClassifier(os.path.expanduser(args.check_model)) \
            if args.check_model and os.path.exists(os.path.expanduser(args.check_model)) else None
        self.warning = ""
        self.corners = None
        self.model = self.detector = None
        self.status = ""
        self.changed = []
        self.prev_small = None
        self.still_since = time.time()
        self.last_check = 0.0
        self.match = False
        self.matches = 0                   # consecutive checks that matched the instruction
        self.moved = False                 # motion seen since the last photo

    # ------------------------------------------------------------ board
    def set_corners(self, corners):
        self.corners = np.asarray(corners, np.float32)
        self.model = BoardModel(self.corners, self.K, self.dist, self.a1_corner)
        self.detector = ChangeDetector(self.model, self.work_shape, threshold=self.threshold)

    def detect_corners(self, work):
        corners, self.a1_corner, msg = locate(work, self.a1_corner)
        print(f"[board] {msg}")
        return corners

    def manual_corners(self, work):
        """Click the outer corners of printed a1, a8, h8, h1."""
        res = click_corners(WIN, work)
        if res is None:
            return None
        corners, self.a1_corner = res
        return corners

    # ------------------------------------------------------------ loop
    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        work = work_image(self.cam.grab())
        corners = self.detect_corners(work) if not self.args.manual else None
        if corners is None:
            corners = self.manual_corners(work)
            if corners is None:
                return
        self.set_corners(corners)
        exp, level = self.cam.auto_expose(measure=self._board_level)
        print(f"[camera] exposure {exp} us (board white level {level:.0f})")

        while True:
            frame = self.cam.grab()
            work = work_image(frame)
            self._track_motion(work)
            ins = self.retake or self.planner.next()
            now = time.time()
            if self.ref is not None and ins.kind != "reset" and self.retake is None \
                    and now - self.last_check > 0.3:
                self.last_check = now
                self.changed, scores = self.detector.changed(self.ref, work)
                self.match, self.status = check_change(ins, self.changed,
                                                       self.detector.allowed(ins.changed))
                self.matches = self.matches + 1 if self.match else 0
            elif ins.kind == "reset" or self.ref is None or self.retake is not None:
                self.match, self.changed, self.matches = False, [], 0
                self.status = "press SPACE when the board looks like the diagram"
            still = now - self.still_since
            # automatic photos only after the human moved something, and when
            # two checks in a row agree (no photo from a single odd frame)
            auto = self.match and self.matches >= 2 and self.moved and still >= STILL_S
            self.draw(work, ins, still)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if key == 32 and ins.kind in ("game", "scatter") and self.retake is None \
                    and not self.moved:
                # nothing moved since the last photo: a second SPACE (or one
                # pressed during the photos) must not label this board as the
                # next position
                self.status = "do the move first (SPACE is only needed if no photo comes)"
            elif key == 32 or auto:
                self.capture(frame, work, ins, "space" if key == 32 else "auto")
            elif key in (ord('n'), ord('N')):
                self.planner.skip()
            elif key in (ord('u'), ord('U')):
                self.undo()
            elif key in (ord('1'), ord('2'), ord('3')):
                new = LIGHTINGS[key - ord('1')]
                if new == self.lighting:
                    continue                # key repeat / pressed again: nothing to do
                self.lighting = new
                print(f"[lighting] {self.lighting}")
                self.ref = None       # everything looks different now: next photo by SPACE
                self.cam.auto_expose(measure=self._board_level)
            elif key in (ord('e'), ord('E')):
                self.cam.auto_expose(measure=self._board_level)
            elif key in (ord('b'), ord('B')):
                corners = self.detect_corners(work)
                if corners is None:
                    corners = self.manual_corners(work)
                if corners is not None:
                    self.set_corners(corners)
            elif key in (ord('+'), ord('=')):
                self.detector.threshold = self.threshold = self.threshold * 1.15
            elif key == ord('-'):
                self.detector.threshold = self.threshold = self.threshold / 1.15
            if isinstance(self.cam, ImageFolderCamera) and key == ord('.'):
                self.cam.next_image()
        self.cam.close()
        cv2.destroyAllWindows()
        print(f"\n{self.photos} positions recorded in {self.dir}")

    def _board_level(self, frame):
        w = work_image(frame)
        mask = np.zeros(w.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, self.corners.astype(np.int32), 1)
        return white_level(w, mask.astype(bool))

    def _track_motion(self, work):
        small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(work, (200, 167)), cv2.COLOR_BGR2GRAY),
                                 (5, 5), 0)
        if self.prev_small is not None:
            moving = (cv2.absdiff(small, self.prev_small) > 25).mean() * 100 > 1.0
            if moving:
                self.still_since = time.time()
                self.moved = True
        self.prev_small = small

    # ------------------------------------------------------------ photos
    def capture(self, frame, work, ins, trigger="space"):
        self.draw(work, ins, 0, banner="TAKING PHOTOS - hands off")
        cv2.waitKey(1)
        corners = find_board_corners(work)[0]
        if corners is not None and np.abs(corners - self.corners).max() < 25:
            self.set_corners(corners)              # follow small shifts of the mat
        level = self._board_level(frame)
        if not 150 <= level <= 230:
            self.cam.adjust(190.0 / max(level, 1.0))
        base = self.cam.exposure_us
        step = self.photos
        after = ins.after
        files = []
        extra = {}
        if ins.kind == "game" and self.planner.game_state is not None:
            board, human_white = self.planner.game_state     # position before the move
            extra = {"uci": ins.move.uci(), "fen_before": board.fen(), "human_white": human_white}
        shots = []
        for k, factor in enumerate(BRACKET):
            exp = self.cam.set_exposure(base * factor)
            shots.append((exp, self.cam.grab_fresh()))
        self.cam.set_exposure(base)
        after_img = self.cam.grab_fresh(flush=3)  # flushes frames with the bracket's last exposure
        # the board must look the same before, at the first exposure and after
        # the series - otherwise a hand came in: throw the photos away
        first = work_image(shots[0][1])
        # (twice the move threshold: a hand scores far higher, lighting noise lower)
        jump = np.maximum(self.detector.scores(work, first),
                          self.detector.scores(first, work_image(after_img)))
        moved = set(np.flatnonzero(jump > 2 * self.threshold).tolist())
        if moved:
            self.status = ("something moved during the photos (" + ", ".join(
                chess.square_name(q) for q in sorted(moved)) + ") - keep hands away, retaking")
            print(f"[retake] board changed during the photos on {sorted(moved)}")
            self.prev_small = None
            self.matches = 0
            self.still_since = time.time()
            return
        with open(self.labels_path, "a") as f:
            for k, (exp, img) in enumerate(shots):
                name = f"{step:05d}_e{k}.jpg"
                cv2.imwrite(os.path.join(self.dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
                files.append(name)
                rec = label_record(name, after, self.corners / WORK_SCALE, exp, self.lighting, step,
                                   ins.text, self.full_size, self.a1_corner, kind=ins.kind,
                                   time=time.time(), gain_db=round(self.cam.gain_db, 2),
                                   trigger=trigger, **extra)
                f.write(json.dumps(rec) + "\n")
        self.ref = first
        self.last_files = files
        self.warning = self._check_label(first, after) if self.checker is not None else ""
        if self.retake is None:
            self.planner.accept()
        self.retake = None
        self.planner.record(after)
        self.photos += 1
        self.prev_small = None
        self.match, self.matches, self.moved = False, 0, False
        self.still_since = time.time()
        self.last_check = time.time()
        for _ in range(20):               # drop keys pressed while the photos were taken
            if cv2.waitKey(5) & 0xFF == 255:
                break
        print(f"[{step:05d}] {ins.text}  ({len(after)} pieces, {self.lighting}, {base} us)")
        if self.photos % LIGHT_CHANGE_EVERY == 0:
            print("[lighting] time to change the lights - then press 1 / 2 / 3")

    def _check_label(self, image, placement):
        """Squares where the classifier is sure the photo shows something
        else than the label - returns the warning text ('' = fine)."""
        probs = self.checker.predict(square_crops(image, self.model))
        bad = []
        for sq in range(64):
            label = CLASSES.index(placement[sq]) if sq in placement else 0
            seen = int(np.argmax(probs[sq]))
            if seen != label and probs[sq, seen] > 0.9 and probs[sq, label] < 0.05:
                bad.append(f"{chess.square_name(sq)}: {_piece_text(seen)}, not {_piece_text(label)}")
        if not bad:
            return ""
        print("[check] " + "; ".join(bad))
        return "CHECK " + "; ".join(bad[:3]) + "  -> fix the board, then U"

    def undo(self):
        files = getattr(self, "last_files", None)
        if not files:
            return
        for name in files:
            try:
                os.remove(os.path.join(self.dir, name))
            except OSError:
                pass
        with open(self.labels_path) as f:
            lines = [ln for ln in f if json.loads(ln)["image"] not in files]
        with open(self.labels_path, "w") as f:
            f.writelines(lines)
        self.photos -= 1
        self.last_files = None
        self.retake = Instruction("Make the board look exactly like the diagram, then SPACE",
                                  dict(self.planner.placement), dict(self.planner.placement),
                                  kind="reset")
        print("[undo] photos deleted - fix the board and press SPACE")

    # ------------------------------------------------------------ drawing
    def draw(self, work, ins, still, banner=None):
        vis = work.copy()
        for sq in self.changed:
            cv2.polylines(vis, [self.model.square_polygon(sq).astype(np.int32)], True,
                          (0, 220, 255), 2)
        for squares, colour in ((ins.sources, (40, 40, 230)), (ins.targets, (40, 190, 40))):
            for sq in squares:
                cv2.polylines(vis, [self.model.square_polygon(sq).astype(np.int32)], True, colour, 4)
        cv2.polylines(vis, [self.corners.astype(np.int32)], True, (255, 160, 0), 1)
        cam = to_display(vis)
        h = 760
        cam = cv2.resize(cam, (int(cam.shape[1] * h / cam.shape[0]), h))
        diag = draw_diagram(ins.after, 560, ins.sources, ins.targets,
                            title="set up this position" if ins.kind == "reset" else "next position")
        W = cam.shape[1] + diag.shape[1] + 30
        canvas = np.full((h + 150, W, 3), 30, np.uint8)
        canvas[:h, :cam.shape[1]] = cam
        canvas[10:10 + diag.shape[0], cam.shape[1] + 20:cam.shape[1] + 20 + diag.shape[1]] = diag
        x0 = cam.shape[1] + 24
        info = [f"photos: {self.photos}   lighting: {self.lighting.upper()} (1 normal 2 dark 3 bright)",
                f"exposure: {self.cam.exposure_us / 1000:.0f} ms +{self.cam.gain_db:.0f} dB   threshold: {self.threshold:.1f} (+/-)",
                f"seen piece/square pairs: {len(self.planner.coverage)} / 768",
                f"still: {min(still, 9.9):.1f} s"]
        for i, t in enumerate(info):
            cv2.putText(canvas, t, (x0, 600 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (220, 220, 220), 1, cv2.LINE_AA)
        colour = (80, 220, 80) if self.match else (0, 200, 255)
        cv2.putText(canvas, ins.text, (12, h + 40), cv2.FONT_HERSHEY_DUPLEX, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, banner or self.status, (12, h + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255) if banner else colour, 2, cv2.LINE_AA)
        if self.warning:
            cv2.putText(canvas, self.warning, (12, h + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
        elif ins.lines:
            cv2.putText(canvas, "  ".join(ins.lines), (12, h + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (180, 180, 180), 1, cv2.LINE_AA)
        if self.photos and self.photos % LIGHT_CHANGE_EVERY == 0:
            cv2.putText(canvas, "Change the lighting now, then press 1 / 2 / 3", (12, h + 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.imshow(WIN, canvas)


def _piece_text(c):
    if c == 0:
        return "empty"
    sym = CLASSES[c]
    return ("white " if sym.isupper() else "black ") + \
        {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}[sym.lower()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="~/chess_dataset")
    ap.add_argument("--resume", help="session directory to continue")
    ap.add_argument("--start", choices=["white", "black"],
                    help="with --resume: begin with setting up the start position "
                         "(white / black on your side)")
    ap.add_argument("--lighting", choices=LIGHTINGS,
                    help="tag stored with every photo (keys 1/2/3 change it); default normal, "
                         "or the last one of a resumed session")
    ap.add_argument("--a1-corner", default="BR", choices=["BL", "TL", "TR", "BR"],
                    help="raw-image corner of printed a1 (BR for the cell's camera)")
    ap.add_argument("--manual", action="store_true", help="click the board corners")
    ap.add_argument("--threshold", type=float, default=9.0,
                    help="change score of a moved square (noise on a still board: ~4)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--check-model", default="~/chess_models/latest/piece_classifier.onnx",
                    help="classifier that checks every photo against its label (if it exists)")
    ap.add_argument("--replay", nargs="+", help="JPEGs instead of the camera ('.' = next image)")
    args = ap.parse_args()
    if args.replay and len(args.replay) == 1 and os.path.isdir(args.replay[0]):
        args.replay = sorted(glob.glob(os.path.join(args.replay[0], "*.jpg")))
    Recorder(args).run()


if __name__ == "__main__":
    main()
