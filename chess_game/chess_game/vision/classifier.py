"""Per-square piece classifier: 64 upright crops -> (64, 13) probabilities
over chess_game.game.CLASSES (empty, PNBRQK, pnbrqk).

Trained with tools/train_classifier.py (PyTorch, any machine with a GPU)
and exported to ONNX.  Here it runs on the fastest backend available (one
board reading = 128 crops, measured on an i7-13700HX / RTX 5060 laptop):
onnxruntime on the GPU (onnxruntime-gpu, ~26 ms), onnxruntime on the CPU
(~150 ms) or OpenCV's DNN module (~240 ms, always there).  The crop
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
    """List of BGR uint8 crops -> N x 3 x H x W float32, normalised RGB
    (OpenCV does the layout / colour swap in C++: ~3x faster than numpy)."""
    blob = cv2.dnn.blobFromImages(list(crops), 1.0 / 255.0, swapRB=True)
    blob -= MEAN.reshape(1, 3, 1, 1)
    blob /= STD.reshape(1, 3, 1, 1)
    return blob


def _ort_session(onnx_path, gpu=True):
    """onnxruntime session on the GPU if possible, else on the CPU; None
    without onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3                     # no warnings about unused providers
    if gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            if hasattr(ort, "preload_dlls"):      # CUDA / cuDNN from the nvidia-* pip packages
                ort.preload_dlls()
            sess = ort.InferenceSession(onnx_path, so, providers=["CUDAExecutionProvider",
                                                                  "CPUExecutionProvider"])
            if "CUDAExecutionProvider" in sess.get_providers():
                return sess
        except Exception as e:  # noqa: BLE001 - broken CUDA install: use the CPU
            print(f"[classifier] GPU not usable ({str(e)[:120]}) - using the CPU")
    return ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])


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
        self.backend = "opencv"
        if prefer in ("auto", "onnxruntime", "cpu"):
            self._ort = _ort_session(onnx_path, gpu=prefer != "cpu")
            if self._ort is None and prefer == "onnxruntime":
                raise RuntimeError("onnxruntime is not available")
        if self._ort is not None:
            self._input = self._ort.get_inputs()[0].name
            self.backend = ("onnxruntime-gpu" if "CUDAExecutionProvider" in self._ort.get_providers()
                            else "onnxruntime-cpu")
        else:
            self.net = cv2.dnn.readNetFromONNX(onnx_path)
        # first run allocates / tunes (cuDNN): do it now, not during the game
        w, h = CROP["size"]
        self.logits([np.zeros((h, w, 3), np.uint8)] * 2)

    def logits(self, crops):
        blob = to_blob(crops)
        if self._ort is not None:
            return self._ort.run(None, {self._input: blob})[0]
        self.net.setInput(blob)
        return self.net.forward()

    def predict(self, crops, tta=True):
        """(N, 13) probabilities (temperature-calibrated if the model's
        metadata has a temperature).  tta: also classify the mirrored crops
        (the model was trained with mirroring) and average."""
        n = len(crops)
        if tta:                     # one batch: the crops and their mirror images
            crops = list(crops) + [np.ascontiguousarray(c[:, ::-1]) for c in crops]
        p = softmax(np.asarray(self.logits(crops), np.float64) / self.temperature)
        return 0.5 * (p[:n] + p[n:]) if tta else p

    def classify(self, image, model, tta=True):
        """Work-size BGR image + BoardModel -> (64, 13) probabilities."""
        return self.predict(square_crops(image, model), tta)
