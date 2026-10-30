#!/usr/bin/env python3
"""Record a labelled photo dataset of the chess pieces - guided.

Run on the robot PC with its display (stop the game nodes first, they hold
the camera):

    source ~/checkers_ws/install/setup.bash
    python3 ~/checkers_ws/src/chess_game/tools/capture_dataset.py --lighting room
    python3 .../capture_dataset.py --resume ~/chess_dataset/<session>
    python3 .../capture_dataset.py --replay a.jpg b.jpg        # no camera

The window shows the camera (the board as you see it from your side), the
position to set up next - red frame = take from, green frame = put on -
and one line saying what to do.  Do it and take your hand away.  When the
board is still and changed exactly as asked, three photos (three exposures)
are taken automatically and the next instruction appears.  Resetting the
board (empty / start position) always waits for SPACE.

Keys:  SPACE  take the photos now (also: confirm a reset)
       N      skip this instruction (e.g. a piece is missing)
       U      the last photos were wrong: delete them, fix the board, SPACE
       L      next lighting tag (do this whenever you change the lights)
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

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from chess_game.dataset import (ChangeDetector, DatasetPlanner, Instruction,  # noqa: E402
                                check_change, label_record, placement_from_fen, read_labels)
from chess_game.ui.diagram import draw_diagram  # noqa: E402
from chess_game.vision.board_model import BoardModel, load_intrinsics  # noqa: E402
from chess_game.vision.camera import (ImageFolderCamera, display_point, raw_point,  # noqa: E402
                                      to_display, white_level)

try:
    from checkers_game.camera.board_geometry import find_board_corners, order_corners
except ImportError:  # running from the source tree without the workspace sourced
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                    "checkers_game"))
    from checkers_game.camera.board_geometry import find_board_corners, order_corners

WIN = "Chess dataset"
CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chess_game", "vision",
                     "ximea_calibration_600.npz")
WORK_SCALE = 0.5                       # detection / change checks on half resolution
BRACKET = (1.0, 0.55, 1.8)             # exposures per position, x the base exposure
LIGHTINGS = ["room", "lamp", "dim", "daylight", "mixed", "side"]
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
            self.lighting = labels[-1]["lighting"] if labels else args.lighting
        else:
            self.dir = os.path.join(os.path.expanduser(args.out), time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(self.dir)
            labels, placement, self.photos, self.lighting = [], {}, 0, args.lighting
        self.labels_path = os.path.join(self.dir, "labels.jsonl")
        with open(os.path.join(self.dir, "session.json"), "w") as f:
            json.dump({"camera": self.cam.serial, "full_size": self.full_size,
                       "calibration": "ximea_calibration_600.npz", "bracket": BRACKET}, f)

        self.planner = DatasetPlanner(placement, seed=args.seed or int(time.time()))
        for r in labels:
            if r["image"].endswith("_e0.jpg"):
                self.planner.record(placement_from_fen(r["placement"]))
        if args.resume:
            self.planner.skip()            # continue scattering from the stored position
        self.ref = None                    # work image of the last photo (change reference)
        self.retake = None                 # Instruction to re-photograph after U
        self.threshold = args.threshold
        self.corners = None
        self.model = self.detector = None
        self.status = ""
        self.changed = []
        self.prev_small = None
        self.still_since = time.time()
        self.last_check = 0.0
        self.match = False

    # ------------------------------------------------------------ board
    def set_corners(self, corners):
        self.corners = np.asarray(corners, np.float32)
        self.model = BoardModel(self.corners, self.K, self.dist, self.a1_corner)
        self.detector = ChangeDetector(self.model, self.work_shape, threshold=self.threshold)

    def detect_corners(self, work):
        corners, msg = find_board_corners(work)
        print(f"[board] {msg}")
        return corners

    def manual_corners(self, work):
        """Click the outer corners of printed a1, a8, h8, h1."""
        names = ["a1", "a8", "h8", "h1"]
        clicks = []

        def on_mouse(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
                clicks.append(raw_point((x / self._disp_scale, y / self._disp_scale),
                                        self.work_shape))
        cv2.setMouseCallback(WIN, on_mouse)
        while len(clicks) < 4:
            disp = to_display(work.copy())
            self._disp_scale = 900.0 / disp.shape[0]
            disp = cv2.resize(disp, None, fx=self._disp_scale, fy=self._disp_scale)
            for i, p in enumerate(clicks):
                d = np.array(display_point(p, self.work_shape)) * self._disp_scale
                cv2.circle(disp, tuple(int(v) for v in d), 6, (0, 0, 255), -1)
            cv2.putText(disp, f"Click the outer corner of the squares at {names[len(clicks)]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(WIN, disp)
            if cv2.waitKey(20) & 0xFF in (27, ord('q')):
                return None
        cv2.setMouseCallback(WIN, lambda *a: None)
        ordered = order_corners(clicks)
        # which image corner (BL, TL, TR, BR) did the user call a1?
        a1 = np.asarray(clicks[0], np.float32)
        self.a1_corner = ("BL", "TL", "TR", "BR")[int(np.argmin(np.linalg.norm(ordered - a1, axis=1)))]
        return ordered

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
            elif ins.kind == "reset" or self.ref is None or self.retake is not None:
                self.match, self.changed = False, []
                self.status = "press SPACE when the board looks like the diagram"
            still = now - self.still_since
            auto = self.match and still >= STILL_S
            self.draw(work, ins, still)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if key == 32 or auto:
                self.capture(frame, work, ins)
            elif key in (ord('n'), ord('N')):
                self.planner.skip()
            elif key in (ord('u'), ord('U')):
                self.undo()
            elif key in (ord('l'), ord('L')):
                self.lighting = LIGHTINGS[(LIGHTINGS.index(self.lighting) + 1) % len(LIGHTINGS)
                                          if self.lighting in LIGHTINGS else 0]
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
        self.prev_small = small

    # ------------------------------------------------------------ photos
    def capture(self, frame, work, ins):
        self.draw(work, ins, 0, banner="TAKING PHOTOS - hands off")
        cv2.waitKey(1)
        corners = find_board_corners(work)[0]
        if corners is not None and np.abs(corners - self.corners).max() < 25:
            self.set_corners(corners)              # follow small shifts of the mat
        level = self._board_level(frame)
        base = self.cam.exposure_us
        if not 150 <= level <= 230:
            base = self.cam.set_exposure(base * 190.0 / max(level, 1.0))
        step = self.photos
        after = ins.after
        files = []
        with open(self.labels_path, "a") as f:
            for k, factor in enumerate(BRACKET):
                exp = self.cam.set_exposure(base * factor)
                img = self.cam.grab_fresh()
                name = f"{step:05d}_e{k}.jpg"
                cv2.imwrite(os.path.join(self.dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
                files.append(name)
                rec = label_record(name, after, self.corners / WORK_SCALE, exp, self.lighting, step,
                                   ins.text, self.full_size, self.a1_corner, kind=ins.kind,
                                   time=time.time())
                f.write(json.dumps(rec) + "\n")
                if k == 0:
                    self.ref = work_image(img)
        self.cam.set_exposure(base)
        self.last_files = files
        if self.retake is None:
            self.planner.accept()
        self.retake = None
        self.planner.record(after)
        self.photos += 1
        self.prev_small = None
        print(f"[{step:05d}] {ins.text}  ({len(after)} pieces, {self.lighting}, {base} us)")
        if self.photos % LIGHT_CHANGE_EVERY == 0:
            print("[lighting] time to change the lights - then press L")

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
        info = [f"photos: {self.photos}   lighting: {self.lighting} (L)",
                f"exposure: {self.cam.exposure_us / 1000:.0f} ms   threshold: {self.threshold:.1f} (+/-)",
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
        if ins.lines:
            cv2.putText(canvas, "  ".join(ins.lines), (12, h + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (180, 180, 180), 1, cv2.LINE_AA)
        if self.photos and self.photos % LIGHT_CHANGE_EVERY == 0:
            cv2.putText(canvas, "Change the lighting now, then press L", (12, h + 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.imshow(WIN, canvas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="~/chess_dataset")
    ap.add_argument("--resume", help="session directory to continue")
    ap.add_argument("--lighting", default="room", help="tag stored with every photo")
    ap.add_argument("--a1-corner", default="BR", choices=["BL", "TL", "TR", "BR"],
                    help="raw-image corner of printed a1 (BR for the cell's camera)")
    ap.add_argument("--manual", action="store_true", help="click the board corners")
    ap.add_argument("--threshold", type=float, default=14.0)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--replay", nargs="+", help="JPEGs instead of the camera ('.' = next image)")
    args = ap.parse_args()
    if args.replay and len(args.replay) == 1 and os.path.isdir(args.replay[0]):
        args.replay = sorted(glob.glob(os.path.join(args.replay[0], "*.jpg")))
    Recorder(args).run()


if __name__ == "__main__":
    main()
