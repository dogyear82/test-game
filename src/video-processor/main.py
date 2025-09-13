import argparse
import base64
import sys
import time
from typing import Optional

import cv2
import requests


def b64_jpeg_from_frame(frame, max_width: int = 640, jpeg_quality: int = 80) -> str:
    # Optionally resize to reduce payload size
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        new_size = (int(w * scale), int(h * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    # Encode to JPEG
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    ok, buf = cv2.imencode('.jpg', frame, encode_params)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")

    return base64.b64encode(buf.tobytes()).decode('utf-8')


def describe_image_with_ollama(
    b64_image: str,
    model: str,
    prompt: str,
    ollama_host: str = "http://localhost:11434",
    timeout: float = 60.0,
) -> Optional[str]:
    url = f"{ollama_host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64_image],
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response")
    except requests.RequestException as e:
        print(f"[error] Ollama request failed: {e}", file=sys.stderr)
        return None
    except ValueError:
        print("[error] Ollama returned non-JSON response", file=sys.stderr)
        return None


def run(
    device: int,
    interval: float,
    model: str,
    prompt: str,
    ollama_host: str,
    width: Optional[int],
    height: Optional[int],
    jpeg_quality: int,
):
    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"[fatal] Unable to open webcam device {device}")
        sys.exit(1)

    print("[info] Webcam opened. Press Ctrl+C to quit.")
    print(f"[info] Using model '{model}' at {ollama_host}")

    last_time = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[warn] Failed to read frame; retrying...")
                time.sleep(0.1)
                continue

            now = time.time()
            if now - last_time < interval:
                # Skip until interval elapses
                continue
            last_time = now

            try:
                b64img = b64_jpeg_from_frame(frame, max_width=640, jpeg_quality=jpeg_quality)
            except Exception as e:
                print(f"[warn] Skipping frame, encode error: {e}")
                continue

            desc = describe_image_with_ollama(
                b64img,
                model=model,
                prompt=prompt,
                ollama_host=ollama_host,
            )
            ts = time.strftime('%H:%M:%S')
            if desc:
                print(f"[{ts}] {desc}")
            else:
                print(f"[{ts}] (no response)")

    except KeyboardInterrupt:
        print("\n[info] Exiting...")
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Stream webcam snapshots to an Ollama vision model and print descriptions."
    )
    parser.add_argument("--device", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between snapshots sent to Ollama (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bakllava",
        help="Ollama model name (e.g., 'llava' or 'llava:13b')",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the scene concisely.",
        help="Prompt sent to the vision model",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server base URL",
    )
    parser.add_argument("--width", type=int, default=None, help="Capture width (optional)")
    parser.add_argument("--height", type=int, default=None, help="Capture height (optional)")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="JPEG quality 1-100 for snapshots (default: 80)",
    )

    args = parser.parse_args()

    run(
        device=args.device,
        interval=args.interval,
        model=args.model,
        prompt=args.prompt,
        ollama_host=args.ollama_host,
        width=args.width,
        height=args.height,
        jpeg_quality=args.jpeg_quality,
    )


if __name__ == "__main__":
    main()

