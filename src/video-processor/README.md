# Video Processor: Webcam to Ollama (Vision)

This simple Python app captures snapshots from your webcam and sends them to an Ollama vision model (e.g., `llava`) for interpretation, printing the model's description to the console.

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally
  - Pull a vision-capable model: `ollama pull llava` (or `llava:13b`)
  - Ensure the server is available at `http://localhost:11434` (default)
- Python dependencies: `pip install -r requirements.txt`
- A working webcam

## Run

From the repo root:

```
python webcam_ollama.py --model llava --interval 5
```

Common options:

- `--device`: Webcam index (default `0`)
- `--interval`: Seconds between snapshots (default `5`)
- `--model`: Ollama model name (e.g., `llava`, `llava:13b`)
- `--prompt`: Prompt for the model (default: concise scene description)
- `--ollama-host`: Ollama base URL (default `http://localhost:11434`)
- `--width` / `--height`: Desired capture size (optional)
- `--jpeg-quality`: JPEG quality for snapshots (default `80`)

Example:

```
python webcam_ollama.py --model llava:13b --interval 3 --prompt "What objects and text do you see?"
```

Press `Ctrl+C` to exit.

## Notes

- The app resizes frames to a max width of 640px before sending to reduce payload size.
- If you get connection errors, confirm Ollama is running: `ollama serve` or open the Ollama app.
- First run may take longer while the model loads.

## Podman (Containerized)

This repo includes a `Containerfile` for Podman.

Build image:

```
podman build -t webcam-ollama -f Containerfile .
```

Run with camera access and point to an Ollama server on the host:

Linux example:

```
podman run --rm -it \
  --device /dev/video0:/dev/video0 \
  -e OLLAMA_HOST=http://host.containers.internal:11434 \
  webcam-ollama --model bakllava --interval 5 --ollama-host $OLLAMA_HOST
```

Alternative (Linux) using host network:

```
podman run --rm -it \
  --network host \
  --device /dev/video0:/dev/video0 \
  webcam-ollama --model llava --interval 5 --ollama-host http://localhost:11434
```

Notes on platforms:

- Camera passthrough (`--device /dev/video0`) works on Linux hosts with V4L2.
- On macOS/Windows, direct webcam access from Linux containers is not generally supported. Prefer running the Python app natively on the host. If you use Podman with WSL2, camera passthrough is typically unavailable.
- To target an Ollama server off-container, set `--ollama-host` to `http://host.containers.internal:11434` (supported on recent Podman) or use `--network host` on Linux.
