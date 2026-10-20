#!/usr/bin/env python3
"""Record a labelled camera dataset for board / piece detection development.

Standalone: needs only the Ximea python API (``ximea.xiapi``), OpenCV and
numpy - no ROS, no GUI.  It runs fine over plain ssh.

    # on the robot PC (stop the checkers_game launch first - camera is exclusive)
    python3 capture_dataset.py                      # full guided session (~10 min)
    python3 capture_dataset.py --quick              # start position only, 2 lightings
    python3 capture_dataset.py --video-only 60      # just record 60 s of play

The script walks you through a list of *scenes* (piece arrangement x
lighting).  For every scene it captures a few frames with several camera
*profiles* (exposure / gain settings) so we can pick the most robust one.

Output (default ~/checkers_captures/<timestamp>/):
    <scene>/<profile>_<n>.png        frame exactly as checkers_node sees it
                                     (resize 600x600 -> undistort -> rot180)
    <scene>/<profile>_raw.jpg        full-resolution raw frame (one per profile)
    video/frame_XXXX.jpg             processed frames of the play video
    meta.json                        camera info + per-frame settings/stats

Afterwards pack it and copy it to your Mac, e.g.
    cd ~ && tar czf checkers_captures.tgz checkers_captures
    scp collab@147.175.115.82:~/checkers_captures.tgz ~/school/cocohrip/src/
"""

import argparse
import datetime as _dt
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

try:
    from ximea import xiapi
except ImportError:  # pragma: no cover - only on the robot PC
    xiapi = None

# --------------------------------------------------------------------------
# camera profiles (what we sweep for each scene)
# --------------------------------------------------------------------------
PROFILES = [
    # name,        exposure_us, gain_db, auto (aeag)
    ("current",    80000,       0.0,     False),   # what checkers_node uses today
    ("exp40k",     40000,       0.0,     False),
    ("exp20k",     20000,       0.0,     False),
    ("auto",       None,        None,    True),    # auto exposure + gain
]

LIGHTINGS = [
    ("L1_room",       "Room lights ON,  desk lamp OFF"),
    ("L2_room_lamp",  "Room lights ON,  desk lamp ON"),
    ("L3_lamp_only",  "Room lights OFF (blinds closed if possible), desk lamp ON"),
    ("L4_dim",        "Room lights OFF, desk lamp OFF (dim - whatever daylight is left)"),
]

ARRANGEMENTS = [
    ("empty", "Remove ALL pieces from the board (board completely empty)."),
    ("start", "Standard START position: robot's BLACK pieces on the 3 rows near the "
              "robot, your WHITE pieces on the 3 rows near you (dark squares only)."),
    ("midgame", "MID-GAME position: move ~6 pieces to the middle of the board, take "
                "~3 pieces of each colour off the board, and if you mark kings "
                "physically (stack / crown) make 1 king of each colour. Put some black "
                "and white pieces on DIAGONALLY-ADJACENT squares. Keep this exact "
                "position for all lightings of this step."),
]


def log(msg=""):
    print(msg, flush=True)


def ask(msg):
    try:
        return input(msg)
    except EOFError:
        return ""


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------
class Camera:
    def __init__(self, calib_path, settle_frames=4):
        if xiapi is None:
            raise RuntimeError("ximea python API not found (from ximea import xiapi)")
        self.cam = xiapi.Camera()
        log("Opening first Ximea camera ...")
        self.cam.open_device()
        self.cam.set_param("imgdataformat", "XI_RGB32")
        self._try("auto_wb", 1)
        self.img = xiapi.Image()
        self.cam.start_acquisition()
        self.settle_frames = settle_frames
        self.calib = None
        if calib_path and os.path.exists(calib_path):
            with np.load(calib_path) as X:
                self.calib = (X["mtx"], X["dist"])
            log(f"Using undistortion from {calib_path}")
        else:
            log(f"WARNING: calibration file not found ({calib_path}); processed "
                "frames will NOT be undistorted")
        self.info = self._camera_info()

    def _try(self, name, value):
        try:
            self.cam.set_param(name, value)
            return True
        except Exception as e:  # noqa: BLE001 - SDK raises its own types
            log(f"  (could not set {name}={value}: {e})")
            return False

    def _get(self, name):
        try:
            return self.cam.get_param(name)
        except Exception:  # noqa: BLE001
            return None

    def _camera_info(self):
        info = {}
        for k in ("device_name", "device_sn", "width", "height"):
            v = self._get(k)
            if isinstance(v, bytes):
                v = v.decode(errors="ignore")
            info[k] = v
        return info

    def apply_profile(self, profile):
        name, exp, gain, auto = profile
        if auto:
            self._try("aeag", 1)
        else:
            self._try("aeag", 0)
            self.cam.set_exposure(int(exp))
            self._try("gain", float(gain))
        # let exposure / auto-exposure converge
        time.sleep(1.5 if auto else 0.3)
        for _ in range(self.settle_frames):
            self.grab_raw()

    def settings(self):
        out = {}
        for k in ("exposure", "gain", "aeag"):
            v = self._get(k)
            out[k] = float(v) if isinstance(v, (int, float)) else v
        return out

    def grab_raw(self):
        self.cam.get_image(self.img)
        return self.img.get_image_data_numpy(invert_rgb_order=False)

    def process(self, raw):
        """Identical to XimeaCamera.get_camera_image()."""
        image = cv2.resize(raw, (600, 600))
        if self.calib is not None:
            mtx, dist = self.calib
            h, w = image.shape[:2]
            newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            image = cv2.undistort(image, mtx, dist, None, newmtx)
            x, y, w, h = roi
            image = image[y:y + h, x:x + w]
        return cv2.rotate(image, cv2.ROTATE_180)

    def close(self):
        try:
            self.cam.stop_acquisition()
            self.cam.close_device()
        except Exception:  # noqa: BLE001
            pass


def frame_stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    return {
        "mean": round(float(gray.mean()), 1),
        "p01": int(np.percentile(gray, 1)),
        "p99": int(np.percentile(gray, 99)),
        "sat_pct": round(float((gray >= 250).mean() * 100), 2),
        "dark_pct": round(float((gray <= 5).mean() * 100), 2),
    }


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if img.ndim == 3 and img.shape[2] == 4 else img


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------
def capture_scene(cam, out_dir, scene, frames_per_profile, meta):
    sdir = os.path.join(out_dir, scene)
    os.makedirs(sdir, exist_ok=True)
    for prof in PROFILES:
        cam.apply_profile(prof)
        for n in range(frames_per_profile):
            raw = cam.grab_raw()
            proc = to_bgr(cam.process(raw))
            fn = f"{prof[0]}_{n}.png"
            cv2.imwrite(os.path.join(sdir, fn), proc)
            if n == 0:
                cv2.imwrite(os.path.join(sdir, f"{prof[0]}_raw.jpg"), to_bgr(raw),
                            [cv2.IMWRITE_JPEG_QUALITY, 93])
            st = frame_stats(proc)
            meta["frames"].append({"scene": scene, "file": f"{scene}/{fn}",
                                   "profile": prof[0], "settings": cam.settings(),
                                   "stats": st, "t": time.time()})
            time.sleep(0.15)
        log(f"    {prof[0]:8s} mean={st['mean']:6.1f}  p1..p99={st['p01']}..{st['p99']}"
            f"  saturated={st['sat_pct']}%  black={st['dark_pct']}%")


def record_video(cam, out_dir, seconds, fps, meta, profile_name="current"):
    prof = next(p for p in PROFILES if p[0] == profile_name)
    cam.apply_profile(prof)
    vdir = os.path.join(out_dir, "video")
    os.makedirs(vdir, exist_ok=True)
    log(f"  Recording {seconds}s at ~{fps} fps with profile '{profile_name}' ...")
    t0 = time.time()
    i = 0
    next_t = t0
    while time.time() - t0 < seconds:
        raw = cam.grab_raw()
        now = time.time()
        if now < next_t:
            continue
        next_t = now + 1.0 / fps
        proc = to_bgr(cam.process(raw))
        cv2.imwrite(os.path.join(vdir, f"frame_{i:04d}.jpg"), proc,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        meta["video"].append({"i": i, "t": round(now - t0, 3)})
        i += 1
        if i % (fps * 5) == 0:
            log(f"    {int(now - t0)} s ...")
    log(f"  Saved {i} frames to {vdir}")


def save_meta(out_dir, meta):
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def default_calib():
    cands = glob.glob(os.path.expanduser(
        "~/*_ws/src/checkers_game/checkers_game/camera/calibration_parameters_ncr_3.npz"))
    cands += glob.glob(os.path.expanduser(
        "~/*_ws/src/*/checkers_game/checkers_game/camera/calibration_parameters_ncr_3.npz"))
    return cands[0] if cands else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.expanduser("~/checkers_captures"))
    ap.add_argument("--calib", default=default_calib(),
                    help="calibration_parameters_ncr_3.npz used by XimeaCamera.undistort")
    ap.add_argument("--frames", type=int, default=2, help="frames per profile per scene")
    ap.add_argument("--quick", action="store_true",
                    help="only start position under L1 and L2")
    ap.add_argument("--video-only", type=int, default=0, metavar="SECONDS")
    ap.add_argument("--video-seconds", type=int, default=60)
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, stamp)
    os.makedirs(out_dir, exist_ok=True)

    cam = Camera(args.calib)
    meta = {"camera": cam.info, "profiles": PROFILES, "calib": args.calib,
            "frames": [], "video": [], "scenes": {}}
    log(f"Camera: {cam.info}")
    log(f"Writing to {out_dir}\n")

    try:
        if args.video_only:
            ask("Press ENTER to start recording the play video ...")
            record_video(cam, out_dir, args.video_only, args.fps, meta)
            return

        if args.quick:
            plan = [(a, l) for a in ARRANGEMENTS if a[0] == "start" for l in LIGHTINGS[:2]]
        else:
            plan = [(a, l) for a in ARRANGEMENTS for l in LIGHTINGS]
        plan_len = len(plan)

        last_arr = None
        for k, ((arr, arr_txt), (light, light_txt)) in enumerate(plan, 1):
            scene = f"{arr}__{light}"
            log("=" * 70)
            log(f"[{k}/{plan_len}] SCENE {scene}")
            if arr != last_arr:
                log(f"  PIECES  : {arr_txt}")
            log(f"  LIGHTING: {light_txt}")
            log("  Keep hands and arm OUT of the camera view.")
            r = ask("  ENTER = capture, s = skip this scene, q = quit > ").strip().lower()
            if r == "q":
                break
            if r == "s":
                continue
            note = ""
            if arr != last_arr:
                note = ask("  Optional note (e.g. 'king at e5', 'lamp angled left'), ENTER to skip > ")
            meta["scenes"][scene] = {"arrangement": arr, "lighting": light, "note": note}
            capture_scene(cam, out_dir, scene, args.frames, meta)
            save_meta(out_dir, meta)
            last_arr = arr

        # hand occlusion
        log("=" * 70)
        log("HAND scene: put the START position back, lighting L2 (room + lamp).")
        log("When you press ENTER, hold your open hand ~15 cm over the middle of the board")
        log("for ~10 s (the script captures while you hold it).")
        if ask("  ENTER = capture, s = skip > ").strip().lower() != "s":
            meta["scenes"]["hand__L2_room_lamp"] = {"arrangement": "start", "lighting": "L2_room_lamp"}
            capture_scene(cam, out_dir, "hand__L2_room_lamp", 1, meta)
            save_meta(out_dir, meta)

        # play video
        log("=" * 70)
        log("VIDEO: start position, lighting L2. When you press ENTER, play ~6 normal moves")
        log("for both sides (move a piece, pull your hand away, wait ~3 s, next move).")
        log("Include at least one capture (remove the jumped piece).")
        if ask(f"  ENTER = record {args.video_seconds}s, s = skip > ").strip().lower() != "s":
            record_video(cam, out_dir, args.video_seconds, args.fps, meta)
    except KeyboardInterrupt:
        log("\nInterrupted - keeping what was recorded.")
    finally:
        save_meta(out_dir, meta)
        cam.close()

    parent = os.path.dirname(out_dir.rstrip("/"))
    log("\nDone. Now pack it:")
    log(f"  cd {parent} && tar czf ~/checkers_captures_{stamp}.tgz {stamp}")
    log("and on your Mac:")
    log(f"  scp collab@<robot-ip>:~/checkers_captures_{stamp}.tgz ~/school/cocohrip/src/")


if __name__ == "__main__":
    sys.exit(main())
