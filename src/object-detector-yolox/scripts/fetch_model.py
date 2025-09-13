import hashlib
import os
from pathlib import Path
import requests


URLS = [
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_nano.onnx",
    "https://raw.githubusercontent.com/onnx/models/main/validated/vision/object_detection_segmentation/yolox/model/yolox_nano.onnx",
    "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/yolox/model/yolox_nano.onnx",
]

DEST_DIR = Path(__file__).resolve().parents[1] / "models"
DEST_PATH = DEST_DIR / "yolox_nano.onnx"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    if DEST_PATH.exists() and DEST_PATH.stat().st_size > 1000:
        print(f"Model already exists at {DEST_PATH} (sha256={sha256(DEST_PATH)[:12]}…)")
        return
    for url in URLS:
        try:
            print(f"Downloading from {url} …")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(DEST_PATH, "wb") as f:
                f.write(r.content)
            print(f"Saved to {DEST_PATH} (sha256={sha256(DEST_PATH)[:12]}…)")
            return
        except Exception as e:
            print(f"Failed: {e}")
    raise SystemExit("Failed to download yolox_nano.onnx from all mirrors.")


if __name__ == "__main__":
    main()

