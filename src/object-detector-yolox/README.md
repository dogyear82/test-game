Object Detector (YOLOX‑nano ONNX) — Webcam Demo

Overview
- Detector: YOLOX‑nano (Apache‑2.0) via ONNX Runtime.
- Hardware: Works on CPU, uses GPU if `onnxruntime-gpu` is available (RTX 3070 Ti supported).
- Input: Default webcam (`--source 0`), or path/RTSP URL.
 - Optional: RTMPose (Apache‑2.0) for pose + prone detection.

Quick Start
- Create venv (recommended):
  - Linux/WSL: `python3 -m venv .venv && source .venv/bin/activate`
  - Windows (PowerShell): `python -m venv .venv; .venv\\Scripts\\Activate.ps1`
- Install deps:
  - Linux: `python -m pip install -r requirements.txt`
  - Windows: `python -m pip install -r requirements.txt` (installs ONNX Runtime DirectML)
- Run: `python app.py --source 0 --imgsz 640 --conf 0.35`
  - Note: The bundled YOLOX‑nano ONNX expects 416×416. The app auto-detects
    and overrides `--imgsz` to 416 when needed.
  - Tracking is enabled by default and labels include a stable `ID`.
  - Pose + prone detection (requires RTMPose ONNX):
    - `python app.py --source 0 --pose --pose-model models/rtmpose_s.onnx --provider cuda`
    - When a person is confirmed prone, their box turns red, label includes [PRONE], and a red PRONE banner is drawn.

Environment Setup (venv + providers)
- Python 3.9+ is recommended. Use a virtual environment to keep deps isolated.
- Install OS-appropriate ONNX Runtime from `requirements.txt`:
  - Linux: `onnxruntime-gpu` (CUDA if available) or falls back to CPU.
  - Windows: `onnxruntime-directml` (D3D12 GPU) or falls back to CPU.
  - macOS: `onnxruntime` CPU.
- Select provider explicitly when needed: `--provider cpu|cuda|dml`.
- On startup the console prints the chosen backend, e.g.: `ONNXRuntime provider: CUDAExecutionProvider`.

Notes
- Providers:
  - Linux: CUDA (if available) → CPU fallback.
  - Windows: DirectML (GPU via D3D12) → CPU fallback.
  - Force a provider: `--provider cpu|cuda|dml`.
- Model: On first setup, place `yolox_nano.onnx` in `models/` or let the included script download it:
  - `python -m scripts.fetch_model`
- Classes: COCO-80 (e.g., person=0, car=2, bus=5, truck=7). Use `--classes` to filter.

Windows Tip
- If `python` opens the Microsoft Store or an "Open with" dialog, either:
  - Use the venv interpreter directly: `.venv\\Scripts\\python.exe ...`
  - Or disable the App Execution Aliases for `python`/`python3` in Windows Settings.

Args
- `--source`: webcam index, video file, or RTSP URL.
- `--imgsz`: network input size (e.g., 640).
- `--conf`: confidence threshold (0–1).
- `--iou`: NMS IoU threshold.
- `--classes`: comma-separated class ids (e.g., `0,2,3,5,7`).
  
Tracking
- The app assigns persistent IDs to detections across frames using a lightweight IOU tracker.
- Flags:
  - `--track-iou`: association IoU threshold (default 0.5).
  - `--track-max-age`: keep unmatched tracks for N frames (default 30).
  - `--track-min-hits`: require N matches before reporting an ID (default 1).
  
Tip: To focus on people only, combine with class filtering: `--classes 0`.

Pose + Prone Detection (RTMPose)
- Download an RTMPose ONNX (COCO‑17 keypoints, input ~256×192) from the MMPose model zoo and place it at `models/rtmpose_s.onnx` (or pass `--pose-model`).
- Enable with `--pose`. The app will crop each tracked person and run RTMPose to estimate keypoints, then detect prone posture from the torso orientation.
- Flags:
  - `--pose-model`: path to RTMPose ONNX.
  - `--pose-interval`: run pose every N frames (default 1).
  - `--prone-angle`: degrees from vertical to classify as prone (default 60).
  - `--prone-frames`: consecutive frames needed to confirm prone (default 5).
  - `--notify-url`: webhook URL to POST JSON when prone is confirmed (default `http://localhost:8080/hook`).
  - `--notify-timeout`: webhook timeout seconds (default 2.5).
  - `--notify-image`: attach a JPEG crop of the prone person as multipart/form-data.
  - `--notify-jpeg-quality`: JPEG quality (1–100) for the attached image (default 85).
  
Notes
- RTMPose is Apache‑2.0 (OK for commercial use). Use `--provider cuda` on Linux with CUDA/cuDNN, or `--provider cpu` if no GPU.
- The current integration supports common RTMPose ONNX outputs: heatmaps or direct keypoints.

License
- YOLOX: Apache-2.0 (commercially permissive). COCO labels are included for convenience.
