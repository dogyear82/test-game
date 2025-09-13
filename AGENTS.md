# Repository Guidelines

## Project Structure & Modules
- `src/object-detector-yolox/`: YOLOX‑nano ONNX webcam demo with optional RTMPose prone detection. Key files: `app.py`, `pose.py`, `tracker.py`, `utils.py`, `models/`.
- `src/video-processor/`: Webcam → Ollama (vision) snapshot streamer. Key files: `main.py`, `README.md`.
- `scripts/`: Utilities, e.g., `webhook_console.py` (simple HTTP receiver), `export-rover-parts.sh`.
- `docs/`: Product and design docs.
- `models/` (root + per‑module): Local model artifacts.

## Build, Test, and Dev Commands
- Object detector (install): `python -m pip install -r src/object-detector-yolox/requirements.txt`
- Run detector (CPU): `python src/object-detector-yolox/app.py --pose --provider cpu`
  - CUDA (Linux): add `--provider cuda`; DirectML (Windows): `--provider dml`.
  - Webhook (default): posts to `http://localhost:8080/hook`; include `--notify-image` to attach a JPEG.
- Fetch YOLOX model: `python -m src.object-detector-yolox.scripts.fetch_model`
- Start webhook console: `python scripts/webhook_console.py --port 8080`
- Video processor: `python src/video-processor/main.py --model llava --interval 5`
- Quick webhook test: `curl -X POST http://localhost:8080/hook -H 'Content-Type: application/json' -d '{"ping":1}'`

## Python Environment Setup
- Prereqs: Python 3.9+ installed (`python --version` or `python3 --version`).
- Create venv in repo root (recommended name: `.venv` and it is git‑ignored):
  - macOS/Linux/WSL: `python3 -m venv .venv`
  - Windows (PowerShell): `py -3 -m venv .venv` or `python -m venv .venv`
- Activate the venv:
  - macOS/Linux/WSL: `source .venv/bin/activate`
  - Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
- Upgrade tooling: `python -m pip install --upgrade pip wheel`
- Install dependencies (pick what you need or both):
  - Detector: `python -m pip install -r src/object-detector-yolox/requirements.txt`
  - Video processor: `python -m pip install -r src/video-processor/requirements.txt`
- Verify: `python -c "import onnxruntime; print(onnxruntime.get_device())"`

Notes
- Do not commit virtualenvs; `.venv/` is excluded via `.gitignore`. If it was accidentally tracked: `git rm -r --cached .venv`.
- On WSL/Windows, avoid staging `.venv/bin/python` symlinks; keeping the venv ignored prevents git errors.
- To pin Python version for tooling, add one of: `.python-version` (pyenv) or `.tool-versions` (asdf), or set `requires-python` in `pyproject.toml`.

## Coding Style & Naming
- Python 3.9+; follow PEP 8 (4‑space indents, snake_case for functions/vars, UpperCamelCase for classes).
- Keep modules cohesive (`app.py` as entrypoints, helpers in dedicated files).
- Prefer explicit, typed function signatures; avoid one‑letter names.
- Paths: use `pathlib.Path`; keep model files under `src/*/models/`.

## Testing Guidelines
- No formal test suite yet. Use manual smoke tests:
  - Detector: run with `--pose --pose-debug`; confirm `[PRONE]` overlay and webhook POST.
  - Webhook: verify receipt with curl (see above).
  - Video processor: confirm periodic descriptions from Ollama.
- Add small repro scripts under `src/*/scripts/` rather than ad‑hoc snippets.

## Commit & PR Guidelines
- Commits: concise, imperative subject; scope prefix when helpful (e.g., `yolox:`, `video:`, `scripts:`). Example: `yolox: add prone webhook default`.
- PRs: include what/why, run instructions, and screenshots/logs where visual output changes. Link related issues. Keep changes minimal and focused.

## Security & Configuration Tips
- Do not commit large model binaries; prefer scripted downloads to `models/`.
- Be mindful of OS‑specific ONNXRuntime providers. On WSL/Windows, verify networking when posting to `localhost`.
