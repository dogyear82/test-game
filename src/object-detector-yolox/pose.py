from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def _select_providers(provider: str):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    provider = (provider or "auto").lower()
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "cuda":
        return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail] or ["CPUExecutionProvider"]
    if provider == "dml":
        return [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in avail] or ["CPUExecutionProvider"]
    # auto priority
    ordered = []
    if "CUDAExecutionProvider" in avail:
        ordered.append("CUDAExecutionProvider")
    if "DmlExecutionProvider" in avail:
        ordered.append("DmlExecutionProvider")
    ordered.append("CPUExecutionProvider")
    return ordered


class RTMPose:
    """Minimal RTMPose (top-down, single-person) ONNX wrapper.

    Expects an ONNX with input [1,3,H,W]. Output supported variants:
      - heatmaps: [1, K, Hh, Wh]
      - keypoints: [1, K, 2] or [1, K, 3] (x,y[,score])
    Coordinates are mapped back to the original image via the bbox crop.
    """

    def __init__(self, onnx_path: Path, provider: str = "auto"):
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"RTMPose model not found: {self.onnx_path}")

        so = ort.SessionOptions()
        so.log_severity_level = 2
        providers = _select_providers(provider)
        self.sess = ort.InferenceSession(str(self.onnx_path), sess_options=so, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        ishape = self.sess.get_inputs()[0].shape
        # [1, 3, H, W]
        self.in_h = int(ishape[2]) if ishape[2] is not None else 256
        self.in_w = int(ishape[3]) if ishape[3] is not None else 192

        # MMPose style normalization
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        # Probe output layout once
        dummy = np.zeros((1, 3, self.in_h, self.in_w), dtype=np.float32)
        out = self.sess.run(None, {self.input_name: dummy})
        self.out = out  # cached for shape inspection

    def _preprocess(self, img_bgr: np.ndarray, bbox_xyxy: np.ndarray, scale: float = 1.25):
        x1, y1, x2, y2 = bbox_xyxy.astype(float)
        w = x2 - x1
        h = y2 - y1
        # expand equally
        cx = x1 + w * 0.5
        cy = y1 + h * 0.5
        half_w = w * 0.5 * scale
        half_h = h * 0.5 * scale
        x1 = max(0, int(round(cx - half_w)))
        y1 = max(0, int(round(cy - half_h)))
        x2 = min(img_bgr.shape[1], int(round(cx + half_w)))
        y2 = min(img_bgr.shape[0], int(round(cy + half_h)))
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((self.in_h, self.in_w, 3), dtype=np.uint8)
            scale_x = (x2 - x1 + 1e-6) / self.in_w
            scale_y = (y2 - y1 + 1e-6) / self.in_h
        else:
            scale_x = (x2 - x1) / self.in_w
            scale_y = (y2 - y1) / self.in_h
        resized = cv2.resize(crop, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (rgb - self.mean) / self.std
        chw = np.transpose(norm, (2, 0, 1))[None, ...]
        meta = {
            "crop_xyxy": (x1, y1, x2, y2),
            "scale_xy": (scale_x, scale_y),
        }
        return chw.astype(np.float32), meta

    @staticmethod
    def _argmax_2d(hm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # hm: [K, H, W] -> coords [K,2], maxvals [K]
        K, H, W = hm.shape
        flat = hm.reshape(K, -1)
        idx = np.argmax(flat, axis=1)
        maxv = flat[np.arange(K), idx]
        ys = (idx // W).astype(np.float32)
        xs = (idx % W).astype(np.float32)
        coords = np.stack([xs, ys], axis=1)
        return coords, maxv, np.array([W, H], dtype=np.float32)

    def _postprocess(self, outputs, meta):
        x1, y1, x2, y2 = meta["crop_xyxy"]
        sx, sy = meta["scale_xy"]

        out = outputs
        if isinstance(out, (list, tuple)):
            out = out[0]

        arr = np.asarray(out)
        # Try common layouts
        if arr.ndim == 4 and arr.shape[0] == 1:
            # heatmaps [1, K, Hh, Wh]
            hm = arr[0]
            coords_hw, maxv, hw = self._argmax_2d(hm)
            # map heatmap -> input -> image
            # scale from heatmap grid to network input size
            Wh, Hh = int(hw[0]), int(hw[1])
            coords_in = np.empty_like(coords_hw)
            coords_in[:, 0] = (coords_hw[:, 0] + 0.5) * (self.in_w / max(1.0, Wh))
            coords_in[:, 1] = (coords_hw[:, 1] + 0.5) * (self.in_h / max(1.0, Hh))
            # map to image
            xs = x1 + coords_in[:, 0] * sx
            ys = y1 + coords_in[:, 1] * sy
            kpts = np.stack([xs, ys, maxv], axis=1)
            return kpts

        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] in (2, 3):
            # keypoints [1, K, 2 or 3] in model input coordinate
            kp = arr[0]
            if kp.shape[1] == 2:
                # attach dummy scores of 1.0
                kp = np.concatenate([kp, np.ones((kp.shape[0], 1), dtype=kp.dtype)], axis=1)
            # kp in model-input pixel coords
            xs = x1 + kp[:, 0] * sx
            ys = y1 + kp[:, 1] * sy
            kpts = np.stack([xs, ys, kp[:, 2]], axis=1)
            return kpts

        # Unknown layout
        raise RuntimeError(f"Unsupported RTMPose output shape: {arr.shape}")

    def infer(self, img_bgr: np.ndarray, bbox_xyxy: np.ndarray):
        inp, meta = self._preprocess(img_bgr, bbox_xyxy)
        out = self.sess.run(None, {self.input_name: inp})
        kpts = self._postprocess(out, meta)
        return kpts  # (K,3) in image coords


def is_prone_by_torso(kpts: np.ndarray, min_conf: float = 0.3, angle_thresh_deg: float = 60.0) -> Tuple[bool, float]:
    """Return (is_prone, angle_deg).

    Uses the angle of the torso vector (mid-shoulders -> mid-hips) relative to vertical.
    Larger angle (~90°) indicates horizontal posture (lying). Requires visible shoulders/hips.
    """
    if kpts is None or kpts.shape[0] < 13:
        return False, 0.0

    # COCO keypoint indices (MMPose order):
    # 5: left shoulder, 6: right shoulder, 11: left hip, 12: right hip
    ls, rs, lh, rh = 5, 6, 11, 12
    needed = [ls, rs, lh, rh]
    if np.any(kpts[needed, 2] < min_conf):
        return False, 0.0

    sh = (kpts[ls, :2] + kpts[rs, :2]) * 0.5
    hp = (kpts[lh, :2] + kpts[rh, :2]) * 0.5
    dv = sh - hp
    dx = float(abs(dv[0]))
    dy = float(abs(dv[1])) + 1e-6
    angle = np.degrees(np.arctan2(dx, dy))  # 0 => vertical, 90 => horizontal
    return (angle >= angle_thresh_deg), angle
