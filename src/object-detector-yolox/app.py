import argparse
import os
import json
from pathlib import Path
import time

import cv2
import numpy as np

from coco_labels import COCO_CLASSES
from utils import letterbox, nms
from tracker import IOUTracker
from pose import RTMPose, is_prone_by_torso


def get_session_and_size(onnx_path: Path, provider: str = "auto"):
    import onnxruntime as ort
    so = ort.SessionOptions()
    # Reduce noisy logs
    so.log_severity_level = 2  # 0=VERBOSE..4=FATAL
    avail = ort.get_available_providers()

    ordered = []
    if provider.lower() == "cpu":
        ordered = ["CPUExecutionProvider"]
    elif provider.lower() == "cuda":
        if "CUDAExecutionProvider" in avail:
            ordered = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            print("CUDA provider not available in this build. Falling back to CPU.")
            ordered = ["CPUExecutionProvider"]
    elif provider.lower() == "dml":
        if "DmlExecutionProvider" in avail:
            ordered = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            print("DirectML provider not available. Falling back to CPU.")
            ordered = ["CPUExecutionProvider"]
    else:  # auto
        if "CUDAExecutionProvider" in avail:
            ordered.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in avail:
            ordered.append("DmlExecutionProvider")
        ordered.append("CPUExecutionProvider")

    try:
        sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=ordered)
        used = sess.get_providers()[0]
        print(f"ONNXRuntime provider: {used}")
    except Exception as e:
        print(f"Failed to initialize provider {ordered}. Falling back to CPU. Error: {e}")
        sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])

    # Infer static input size if present
    ishape = sess.get_inputs()[0].shape  # [1, 3, H, W] or dynamic
    h = ishape[2] if isinstance(ishape[2], int) else None
    w = ishape[3] if isinstance(ishape[3], int) else None
    return sess, (h, w)


def make_grids(img_h, img_w, strides=(8, 16, 32)):
    grids = []
    expanded_strides = []
    for s in strides:
        gh, gw = img_h // s, img_w // s
        xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((1, gh * gw, 1), s))
    grids = np.concatenate(grids, axis=1)
    expanded_strides = np.concatenate(expanded_strides, axis=1)
    return grids, expanded_strides


def postprocess(prediction, img_size, conf_thres=0.3, iou_thres=0.5, classes=None):
    # prediction: (1, N, 85) raw YOLOX output
    # Decode bboxes
    img_h, img_w = img_size
    grids, expanded_strides = make_grids(img_h, img_w)

    pred = prediction[0]
    # Sigmoid on obj and cls
    obj = 1.0 / (1.0 + np.exp(-pred[:, 4:5]))
    cls = 1.0 / (1.0 + np.exp(-pred[:, 5:]))
    scores = obj * cls
    cls_ids = scores.argmax(axis=1)
    conf = scores.max(axis=1)

    # Decode boxes
    # x,y: (raw + grid) * stride; w,h: exp(raw) * stride
    xy = (pred[:, 0:2] + grids.reshape(-1, 2)) * expanded_strides.reshape(-1, 1)
    wh = np.exp(pred[:, 2:4]) * expanded_strides.reshape(-1, 1)
    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2
    boxes = np.concatenate([x1y1, x2y2], axis=1)

    # Filter by conf and classes
    m = conf > conf_thres
    if classes is not None:
        m = np.logical_and(m, np.isin(cls_ids, np.array(classes)))
    boxes, conf, cls_ids = boxes[m], conf[m], cls_ids[m]

    # NMS per class
    keep_indices = []
    final_boxes, final_scores, final_cls = [], [], []
    for c in np.unique(cls_ids):
        idxs = np.where(cls_ids == c)[0]
        if idxs.size == 0:
            continue
        k = nms(boxes[idxs], conf[idxs], iou_thres)
        keep = idxs[k]
        keep_indices.extend(keep.tolist())
    if keep_indices:
        final_boxes = boxes[keep_indices]
        final_scores = conf[keep_indices]
        final_cls = cls_ids[keep_indices]
    else:
        final_boxes = np.zeros((0, 4), dtype=np.float32)
        final_scores = np.zeros((0,), dtype=np.float32)
        final_cls = np.zeros((0,), dtype=np.int32)

    return final_boxes, final_scores, final_cls


def infer_frame(session, frame_bgr, imgsz=640, conf=0.35, iou=0.5, classes=None):
    h0, w0 = frame_bgr.shape[:2]
    img, r, (dw, dh) = letterbox(frame_bgr, imgsz)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    img_chw = img_chw.astype(np.float32) / 255.0
    inp = np.expand_dims(img_chw, 0)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inp})
    pred = outputs[0] if isinstance(outputs, list) else outputs
    # Expect (1, N, 85)
    boxes, scores, cls_ids = postprocess(pred, (img.shape[0], img.shape[1]), conf_thres=conf, iou_thres=iou, classes=classes)

    # Map boxes back to original image space
    if boxes.shape[0] > 0:
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes[:, :4] /= r
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, w0)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, h0)

    return boxes, scores, cls_ids


def draw_dets(img, boxes, scores, cls_ids, names, ids=None, prone_ids=None, color=(0, 255, 0)):
    for i, ((x1, y1, x2, y2), s, c) in enumerate(zip(boxes, scores, cls_ids)):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        det_id = int(ids[i]) if ids is not None and len(ids) > i else None
        is_prone = prone_ids is not None and det_id is not None and det_id in prone_ids
        box_color = (0, 0, 255) if is_prone else color
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        id_prefix = f"ID {det_id} " if det_id is not None else ""
        label = f"{id_prefix}{names[c]} {s:.2f}"
        if is_prone:
            label += " [PRONE]"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # If prone, draw a red tag above the main label
        if is_prone:
            tag = "PRONE"
            (pw, ph), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_top = max(0, y1 - th - ph - 12)
            cv2.rectangle(img, (x1, y_top), (x1 + pw + 8, y_top + ph + 6), (0, 0, 255), -1)
            cv2.putText(img, tag, (x1 + 4, y_top + ph), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='0', help='Webcam index (e.g., 0) or video path/RTSP URL')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--classes', type=str, default=None, help='Comma-separated class IDs to keep')
    ap.add_argument('--model', type=str, default=str(Path(__file__).resolve().parent / 'models' / 'yolox_nano.onnx'))
    ap.add_argument('--provider', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'dml'], help='Inference provider preference')
    # Tracking
    ap.add_argument('--track-iou', type=float, default=0.5, help='IOU threshold for association')
    ap.add_argument('--track-max-age', type=int, default=30, help='Frames to keep lost tracks')
    ap.add_argument('--track-min-hits', type=int, default=1, help='Min matches before reporting ID')
    # Pose / prone detection
    ap.add_argument('--pose', action='store_true', help='Enable RTMPose pose estimation + prone detection')
    ap.add_argument('--pose-model', type=str, default=str(Path(__file__).resolve().parent / 'models' / 'rtmpose_s.onnx'))
    ap.add_argument('--pose-interval', type=int, default=1, help='Run pose on each track every N frames')
    ap.add_argument('--prone-angle', type=float, default=60.0, help='Angle threshold (deg) for prone detection')
    ap.add_argument('--prone-frames', type=int, default=5, help='Consecutive frames to confirm prone')
    ap.add_argument('--pose-debug', action='store_true', help='Print RTMPose/prone debug info each evaluation')
    # Notifications
    ap.add_argument('--notify-url', type=str, default='http://localhost:8080/hook', help='Webhook URL to POST a JSON event on prone confirmation')
    ap.add_argument('--notify-timeout', type=float, default=2.5, help='Webhook timeout seconds')
    ap.add_argument('--notify-image', action='store_true', help='Attach JPEG crop of the prone person in webhook (multipart/form-data)')
    ap.add_argument('--notify-jpeg-quality', type=int, default=85, help='JPEG quality for webhook image (1-100)')
    return ap.parse_args()


def main():
    args = parse_args()
    classes = None
    if args.classes:
        classes = [int(x.strip()) for x in args.classes.split(',') if x.strip()]

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found at {model_path}. Try: python -m scripts.fetch_model")
        return

    session, static_hw = get_session_and_size(model_path, provider=args.provider)
    # Honor static network size if the ONNX model is fixed-size (e.g., 416)
    net_h, net_w = static_hw
    if net_h and net_w:
        if args.imgsz != net_h:
            print(f"Model has fixed input {net_h}x{net_w}. Overriding --imgsz {args.imgsz} -> {net_h}.")
        args.imgsz = net_h
    names = COCO_CLASSES

    # Open source
    source = args.source
    cap_index = None
    if source.isdigit():
        cap_index = int(source)
        cap = cv2.VideoCapture(cap_index)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise SystemExit(f"Failed to open source: {source}")

    t0 = time.time()
    frames = 0

    # Initialize tracker
    tracker = IOUTracker(iou_threshold=args.track_iou, max_age=args.track_max_age, min_hits=args.track_min_hits)
    # Track which ball IDs have been announced
    seen_ball_ids = set()
    # COCO class id for "sports ball" (print notification when seen)
    try:
        ball_cls_id = COCO_CLASSES.index("sports ball")
    except ValueError:
        ball_cls_id = None
    # Pose / prone state
    rtmpose = None
    prone_counts = {}  # id -> consecutive prone frames
    prone_announced = set()  # ids already notified
    prone_active = set()  # ids currently considered prone
    person_cls_id = 0  # COCO person
    if args.pose:
        try:
            rtmpose = RTMPose(Path(args.pose_model), provider=args.provider)
            print("RTMPose initialized.")
        except Exception as e:
            print(f"Pose disabled: failed to load RTMPose model: {e}")
            rtmpose = None

    def notify_prone(track_id: int, angle: float, frame, bbox):
        ts = time.strftime('%Y-%m-%dT%H:%M:%S')
        msg = f"[{ts}] Track ID {int(track_id)} is prone (angle {angle:.1f}°)"
        print(msg)
        if args.notify_url:
            try:
                import requests
                payload = {
                    'ts': ts,
                    'event': 'prone_confirmed',
                    'track_id': int(track_id),
                    'angle_deg': float(angle),
                }
                if args.notify_image and frame is not None and bbox is not None:
                    # Crop person bbox and attach as JPEG
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
                    x1 = min(x1, w - 1)
                    x2 = min(x2, w)
                    y1 = min(y1, h - 1)
                    y2 = min(y2, h)
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                    else:
                        crop = frame
                    ok, enc = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(100, args.notify_jpeg_quality)))])
                    if ok:
                        files = {
                            'image': ('prone.jpg', enc.tobytes(), 'image/jpeg')
                        }
                        data = {'payload': json.dumps(payload)}
                        requests.post(args.notify_url, data=data, files=files, timeout=args.notify_timeout)
                    else:
                        # Fallback to JSON-only
                        requests.post(args.notify_url, json=payload, timeout=args.notify_timeout)
                else:
                    requests.post(args.notify_url, json=payload, timeout=args.notify_timeout)
            except Exception as e:
                print(f"Webhook notify failed: {e}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        boxes, scores, cls_ids = infer_frame(session, frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, classes=classes)

        # Update tracker and draw with stable IDs
        t_boxes, t_scores, t_cls, t_ids = tracker.update(boxes, scores, cls_ids)
        out = draw_dets(frame.copy(), t_boxes, t_scores, t_cls, names, ids=t_ids, prone_ids=prone_active)

        # Pose + prone detection (once per pose-interval)
        if rtmpose is not None and len(t_ids) > 0 and frames % max(1, args.pose_interval) == 0:
            # Evaluate prone state for visible person tracks
            for (tb, tc, tid) in zip(t_boxes, t_cls, t_ids):
                tid = int(tid)
                if int(tc) != person_cls_id:
                    continue
                try:
                    kpts = rtmpose.infer(frame, tb)
                except Exception as e:
                    if args.pose_debug:
                        import traceback
                        print(f"Pose error on track {tid}: {e}")
                        traceback.print_exc()
                    continue
                is_prone, angle = is_prone_by_torso(kpts, min_conf=0.3, angle_thresh_deg=args.prone_angle)
                if args.pose_debug:
                    vis = int((kpts[:,2] >= 0.3).sum()) if kpts is not None else 0
                    print(f"Pose dbg: track {tid} angle={angle:.1f} deg, prone={is_prone}, visible_kpts>0.3={vis}")
                if is_prone:
                    prone_counts[tid] = prone_counts.get(tid, 0) + 1
                    if prone_counts[tid] >= args.prone_frames:
                        # Mark active and notify once
                        if tid not in prone_active:
                            prone_active.add(tid)
                        if tid not in prone_announced:
                            notify_prone(tid, angle, frame, tb)
                            prone_announced.add(tid)
                else:
                    prone_counts[tid] = 0
                    if tid in prone_active:
                        prone_active.discard(tid)

        # One-time console notification for newly seen balls
        if ball_cls_id is not None and len(t_ids) > 0:
            for cls_i, tid in zip(t_cls, t_ids):
                if int(cls_i) == ball_cls_id and int(tid) not in seen_ball_ids:
                    ts = time.strftime('%H:%M:%S')
                    print(f"[{ts}] Detected ball with ID {int(tid)}")
                    seen_ball_ids.add(int(tid))
        dt = (time.time() - t0)
        fps = frames / dt if dt > 0 else 0.0
        cv2.putText(out, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 220, 50), 2)
        cv2.imshow('YOLOX-nano ONNX (Webcam)', out)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
