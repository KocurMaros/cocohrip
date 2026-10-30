"""Per-square piece classifier: 64 upright crops -> (64, 13) probabilities
over chess_game.game.CLASSES (empty, PNBRQK, pnbrqk).

Trained with tools/train_classifier.py (PyTorch, any machine with a GPU)
and exported to ONNX; here it runs with OpenCV's DNN module (no torch
needed on the robot PC), or onnxruntime if it is installed.  The crop
geometry and the input normalisation live in this file so training and
inference cannot drift apart.
"""

import json
import os

import cv2
import numpy as np

# crop geometry (BoardModel.column_crop arguments) - the model is trained on these
CROP = dict(size=(96, 160), height=1.8, below=0.65, width=1.3, span=2.2)
# frames are classified at half the sensor resolution (2464x2056 -> 1232x1028)
WORK_SCALE = 0.5
MEAN = np.array([0.485, 0.456, 0.406], np.float32)     # ImageNet, RGB
STD = np.array([0.229, 0.224, 0.225], np.float32)


def work_image(frame):
    return cv2.resize(frame, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)


def square_crops(image, model, **overrides):
    """64 BGR crops (printed square order a1..h8) of a work-size image."""
    kw = dict(CROP, **overrides)
    return [model.column_crop(image, sq, **kw) for sq in range(64)]


def to_blob(crops):
    """List of BGR uint8 crops -> N x 3 x H x W float32, normalised RGB."""
    x = np.stack(crops).astype(np.float32)[..., ::-1] / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SquareClassifier:
    def __init__(self, onnx_path, prefer="auto"):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"no piece classifier at {onnx_path} - train one with "
                                    "chess_game/tools/train_classifier.py")
        self.path = onnx_path
        meta_path = os.path.splitext(onnx_path)[0] + ".json"
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        self.temperature = float(self.meta.get("temperature", 1.0))
        self._ort = None
        if prefer in ("auto", "onnxruntime"):
            try:
                import onnxruntime as ort
                self._ort = ort.InferenceSession(onnx_path, providers=[
                    p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                    if p in ort.get_available_providers()])
                self._input = self._ort.get_inputs()[0].name
            except ImportError:
                if prefer == "onnxruntime":
                    raise
        if self._ort is None:
            self.net = cv2.dnn.readNetFromONNX(onnx_path)

    def logits(self, crops):
        blob = to_blob(crops)
        if self._ort is not None:
            return self._ort.run(None, {self._input: blob})[0]
        self.net.setInput(blob)
        return self.net.forward()

    def predict(self, crops):
        """(N, 13) probabilities (temperature-calibrated if the model's
        metadata has a temperature)."""
        return softmax(np.asarray(self.logits(crops), np.float64) / self.temperature)

    def classify(self, image, model):
        """Work-size BGR image + BoardModel -> (64, 13) probabilities."""
        return self.predict(square_crops(image, model))
