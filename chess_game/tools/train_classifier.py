#!/usr/bin/env python3
"""Train the per-square piece classifier on sessions recorded with
capture_dataset.py and export it to ONNX.

    python3 train_classifier.py ~/chess_dataset/2026* --out ~/chess_models/v1
    python3 train_classifier.py ... --arch mobilenet_v3_large --epochs 30
    # new piece set: fine-tune the existing model on a short session
    python3 train_classifier.py ~/chess_dataset/<new set>* --init ~/chess_models/latest/piece_classifier.pt \
        --lr 1e-4 --epochs 10 --out ~/chess_models/<new set>

Needs PyTorch + torchvision (a GPU makes it ~20x faster); everything else
is the chess_game package.  Labels come for free: every photo has the FEN of
the physical board and the board corners, so the 64 crops and their classes
follow from the 3-D board model.  Validation holds out whole blocks of
consecutive positions (the three exposures of a position and its
neighbours look almost the same).

Output: <out>/piece_classifier.onnx + piece_classifier.json (classes, crop
geometry, temperature for calibrated probabilities, validation metrics).
"""

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from chess_game.dataset import placement_from_fen, read_labels  # noqa: E402
from chess_game.game import CLASSES  # noqa: E402
from chess_game.vision.board_model import BoardModel, load_intrinsics  # noqa: E402
from chess_game.vision.classifier import CROP, MEAN, STD, WORK_SCALE  # noqa: E402

CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chess_game", "vision",
                     "ximea_calibration_600.npz")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_records(session_dirs, val_block=25, val_every=5):
    """All photos of the sessions; split into train / val by blocks of
    `val_block` consecutive positions (every `val_every`-th block is val)."""
    train, val = [], []
    for d in session_dirs:
        path = os.path.join(d, "labels.jsonl")
        if not os.path.exists(path):
            print(f"skip {d}: no labels.jsonl")
            continue
        for r in read_labels(path):
            r["path"] = os.path.join(d, r["image"])
            r["session"] = os.path.basename(os.path.normpath(d))
            (val if (r["step"] // val_block) % val_every == val_every - 1 else train).append(r)
    return train, val


def square_labels(rec):
    placement = placement_from_fen(rec["placement"])
    return np.array([CLASSES.index(placement[s]) if s in placement else 0 for s in range(64)])


def read_work(path):
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)       # half resolution on decode
    return img


class PhotoSquares(torch.utils.data.Dataset):
    """One item = one photo: its crops (all pieces + `empties` random empty
    squares when training, all 64 for validation) and their labels."""

    def __init__(self, records, train, empties=12, corner_jitter=2.0):
        self.records = records
        self.train = train
        self.empties = empties
        self.corner_jitter = corner_jitter
        self._intr = {}

    def __len__(self):
        return len(self.records)

    def _model(self, rec, shape):
        key = (shape[1], shape[0])
        if key not in self._intr:
            self._intr[key] = load_intrinsics(CALIB, key)
        K, dist = self._intr[key]
        corners = np.asarray(rec["corners"], np.float64) * WORK_SCALE
        if self.train and self.corner_jitter:
            corners = corners + np.random.uniform(-self.corner_jitter, self.corner_jitter, corners.shape)
        return BoardModel(corners, K, dist, rec.get("a1_corner", "BR"))

    def __getitem__(self, i):
        rec = self.records[i]
        img = read_work(rec["path"])
        labels = square_labels(rec)
        squares = list(range(64))
        if self.train:
            occupied = [s for s in squares if labels[s]]
            empty = [s for s in squares if not labels[s]]
            squares = occupied + random.sample(empty, min(self.empties, len(empty)))
        model = self._model(rec, img.shape)
        crops = [model.column_crop(img, s, **CROP) for s in squares]
        if self.train:
            crops = [augment(c) for c in crops]
        x = np.stack(crops).astype(np.float32)[..., ::-1] / 255.0
        x = (x - MEAN) / STD
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))),
                torch.from_numpy(labels[squares]).long(),
                torch.full((len(squares),), i, dtype=torch.long))


def collate(batch):
    xs, ys, ids = zip(*batch)
    return torch.cat(xs), torch.cat(ys), torch.cat(ids)


def augment(crop):
    """Lighting (exposure, contrast, gamma, white balance, noise, blur) and
    small geometric jitter (board corners / piece off-centre)."""
    img = crop.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    # geometry: shift, rotate, scale around the piece base
    if random.random() < 0.8:
        ang = random.uniform(-6, 6)
        sc = random.uniform(0.92, 1.08)
        M = cv2.getRotationMatrix2D((w / 2, h * 0.7), ang, sc)
        M[:, 2] += (random.uniform(-4, 4), random.uniform(-4, 4))
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.5:
        img = img[:, ::-1]
    # photometric
    img = img * np.exp(random.gauss(0, 0.25))                           # exposure
    img = (img - 0.5) * np.exp(random.gauss(0, 0.15)) + 0.5             # contrast
    img = img * np.exp(np.random.normal(0, 0.06, 3)).astype(np.float32)  # white balance
    img = np.clip(img, 0, 1) ** np.exp(random.gauss(0, 0.2))            # gamma
    if random.random() < 0.3:
        k = random.choice((3, 5))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.5:
        img = img + np.random.normal(0, random.uniform(0.005, 0.04), img.shape).astype(np.float32)
    if random.random() < 0.2:                                           # shadow / highlight band
        y0 = random.randint(0, h - 1)
        img[y0:y0 + random.randint(5, h // 2)] *= random.uniform(0.5, 1.4)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def build_model(arch, n_classes=len(CLASSES), pretrained=True):
    if arch == "resnet18":
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == "mobilenet_v3_large":
        m = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2" if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif arch == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    else:
        raise ValueError(arch)
    return m


@torch.no_grad()
def evaluate(model, loader, device, records):
    model.eval()
    all_logits, all_y, all_ids = [], [], []
    for x, y, ids in loader:
        all_logits.append(model(x.to(device)).float().cpu())
        all_y.append(y)
        all_ids.append(ids)
    logits, y, ids = torch.cat(all_logits), torch.cat(all_y), torch.cat(all_ids)
    pred = logits.argmax(1)
    ok = (pred == y)
    per_class = {CLASSES[c] if c else "empty": float(ok[y == c].float().mean())
                 for c in range(len(CLASSES)) if (y == c).any()}
    boards = defaultdict(list)
    for i, good in zip(ids.tolist(), ok.tolist()):
        boards[i].append(good)
    by_light = defaultdict(list)
    for i, goods in boards.items():
        by_light[records[i]["lighting"]].append(all(goods))
    return {
        "square_acc": float(ok.float().mean()),
        "piece_acc": float(ok[y > 0].float().mean()) if (y > 0).any() else 0.0,
        "board_acc": float(np.mean([all(g) for g in boards.values()])),
        "board_acc_by_lighting": {k: float(np.mean(v)) for k, v in by_light.items()},
        "per_class": per_class,
        "worst_photos": sorted(((sum(not g for g in goods), records[i]["image"], records[i]["session"])
                                for i, goods in boards.items()), reverse=True)[:15],
    }, logits, y


def fit_temperature(logits, y):
    """Temperature scaling on validation logits (calibrated probabilities
    make the move matcher's likelihoods meaningful)."""
    t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)
    lossf = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = lossf(logits / t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.exp())


def export_onnx(model, path, device):
    model.eval()
    w, h = CROP["size"]
    dummy = torch.zeros(2, 3, h, w, device=device)
    kw = dict(input_names=["crops"], output_names=["logits"],
              dynamic_axes={"crops": {0: "n"}, "logits": {0: "n"}}, opset_version=13)
    try:   # torch >= 2.9 defaults to the dynamo exporter (needs onnxscript)
        torch.onnx.export(model, dummy, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, path, **kw)
    # check OpenCV can run it and agrees with PyTorch
    net = cv2.dnn.readNetFromONNX(path)
    x = torch.randn(3, 3, h, w)
    net.setInput(x.numpy())
    ref = model(x.to(device)).float().detach().cpu().numpy()
    diff = float(np.abs(net.forward() - ref).max())
    print(f"ONNX export {path}: OpenCV vs PyTorch max |diff| {diff:.2e}")
    return diff


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="session directories (globs ok)")
    ap.add_argument("--out", default="~/chess_models/latest")
    ap.add_argument("--arch", default="resnet18",
                    choices=["resnet18", "mobilenet_v3_large", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--photos-per-batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--init", help="start from this piece_classifier.pt (fine-tuning, e.g. a "
                                   "new piece set - use a lower --lr such as 1e-4)")
    ap.add_argument("--max-photos", type=int, help="use at most N photos (quick tests)")
    ap.add_argument("--val-block", type=int, default=25,
                    help="every 5th block of this many consecutive positions is validation")
    args = ap.parse_args()

    sessions = sorted({p for s in args.sessions for p in glob.glob(os.path.expanduser(s))})
    train_recs, val_recs = load_records(sessions, val_block=args.val_block)
    if args.max_photos:
        train_recs, val_recs = train_recs[:args.max_photos], val_recs[:max(1, args.max_photos // 4)]
    if not train_recs or not val_recs:
        sys.exit("need photos for both training and validation")
    print(f"{len(sessions)} sessions, {len(train_recs)} train / {len(val_recs)} val photos, "
          f"lighting {dict(Counter(r['lighting'] for r in train_recs + val_recs))}")

    # class weights ~ 1/sqrt(frequency) of the sampled crops
    counts = Counter()
    for r in train_recs:
        lab = square_labels(r)
        counts.update(lab[lab > 0].tolist())
        counts[0] += min(12, int((lab == 0).sum()))
    freq = np.array([max(counts[c], 1) for c in range(len(CLASSES))], np.float64)
    weights = torch.tensor((freq.sum() / freq) ** 0.5, dtype=torch.float32)
    weights /= weights.mean()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch)
    if args.init:
        model.load_state_dict(torch.load(os.path.expanduser(args.init), map_location="cpu"))
        print(f"fine-tuning from {args.init}")
    model = model.to(device)
    train_dl = torch.utils.data.DataLoader(PhotoSquares(train_recs, True), args.photos_per_batch,
                                           shuffle=True, num_workers=args.workers, collate_fn=collate,
                                           persistent_workers=args.workers > 0, drop_last=True)
    val_dl = torch.utils.data.DataLoader(PhotoSquares(val_recs, False), args.photos_per_batch,
                                         num_workers=args.workers, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train_dl),
                                                pct_start=0.15)
    lossf = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    best, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = lossf(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        metrics, _, _ = evaluate(model, val_dl, device, val_recs)
        print(f"epoch {epoch + 1:2d}/{args.epochs}  loss {total / max(n, 1):.4f}  "
              f"val square {metrics['square_acc']:.4f}  piece {metrics['piece_acc']:.4f}  "
              f"board {metrics['board_acc']:.3f}  ({time.time() - t0:.0f} s)")
        if metrics["square_acc"] > best:
            best = metrics["square_acc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    metrics, logits, y = evaluate(model, val_dl, device, val_recs)
    temperature = fit_temperature(logits, y)
    torch.save(best_state, os.path.join(out, "piece_classifier.pt"))
    diff = export_onnx(model, os.path.join(out, "piece_classifier.onnx"), device)
    meta = {"classes": CLASSES, "arch": args.arch, "crop": CROP, "work_scale": WORK_SCALE,
            "temperature": temperature, "val": metrics, "onnx_opencv_diff": diff,
            "sessions": [os.path.basename(s) for s in sessions],
            "photos": {"train": len(train_recs), "val": len(val_recs)},
            "trained": time.strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(out, "piece_classifier.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: metrics[k] for k in ("square_acc", "piece_acc", "board_acc",
                                              "board_acc_by_lighting", "per_class")}, indent=1))
    print("worst validation photos (wrong squares, photo, session):")
    for row in metrics["worst_photos"][:8]:
        print("  ", row)
    print(f"temperature {temperature:.2f} -> {out}/piece_classifier.onnx")


if __name__ == "__main__":
    main()
