import numpy as np

from utils import iou as iou_fn


class Track:
    def __init__(self, bbox: np.ndarray, score: float, cls_id: int, track_id: int):
        self.bbox = bbox.astype(np.float32)
        self.score = float(score)
        self.cls_id = int(cls_id)
        self.id = int(track_id)

        self.hits = 1  # total matches
        self.age = 0  # total frames since created
        self.time_since_update = 0  # frames since last match

        cx = 0.5 * (self.bbox[0] + self.bbox[2])
        cy = 0.5 * (self.bbox[1] + self.bbox[3])
        self.last_center = np.array([cx, cy], dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)

    def update(self, bbox: np.ndarray, score: float):
        # Update bbox and basic velocity estimate
        bbox = bbox.astype(np.float32)
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        center = np.array([cx, cy], dtype=np.float32)
        self.velocity = center - self.last_center
        self.last_center = center
        self.bbox = bbox
        self.score = float(score)
        self.hits += 1
        self.time_since_update = 0


class IOUTracker:
    """
    Minimal IOU-based tracker assigning stable IDs across frames.
    - Greedy matching per class by highest IoU.
    - New detections spawn new tracks.
    - Tracks are removed if not seen for `max_age` frames.
    """

    def __init__(self, iou_threshold: float = 0.5, max_age: int = 30, min_hits: int = 1, track_classes=None):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.track_classes = set(track_classes) if track_classes is not None else None

        self.tracks = []  # type: list[Track]
        self.next_id = 1

    def _add_track(self, bbox: np.ndarray, score: float, cls_id: int):
        t = Track(bbox, score, int(cls_id), self.next_id)
        self.next_id += 1
        self.tracks.append(t)

    def _prune(self):
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

    def update(self, boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray):
        """
        Update tracker with current detections.
        Returns boxes, scores, cls_ids, ids for tracks updated this frame.
        """
        # Age all tracks by default; matched ones will reset time_since_update
        for t in self.tracks:
            t.time_since_update += 1
            t.age += 1

        # Nothing detected
        if boxes is None or len(boxes) == 0:
            self._prune()
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )

        boxes = boxes.astype(np.float32)
        scores = scores.astype(np.float32)
        cls_ids = cls_ids.astype(np.int32)

        if self.track_classes is not None:
            keep = np.isin(cls_ids, np.array(list(self.track_classes), dtype=np.int32))
            boxes_in = boxes[keep]
            scores_in = scores[keep]
            cls_in = cls_ids[keep]
        else:
            boxes_in, scores_in, cls_in = boxes, scores, cls_ids

        # Per-class greedy matching
        matched_det = set()
        matched_trk = set()
        assignments = []  # (trk_index, det_index)

        unique_classes = set(cls_in.tolist()) | set([t.cls_id for t in self.tracks])
        for c in unique_classes:
            det_idx = np.where(cls_in == c)[0]
            trk_idx = [i for i, t in enumerate(self.tracks) if t.cls_id == c]
            if len(det_idx) == 0 or len(trk_idx) == 0:
                continue

            det_boxes = boxes_in[det_idx]
            trk_boxes = np.stack([self.tracks[i].bbox for i in trk_idx], axis=0)

            # IoU matrix (T, D)
            iou_mat = np.zeros((len(trk_idx), len(det_idx)), dtype=np.float32)
            for ti in range(len(trk_idx)):
                iou_mat[ti] = iou_fn(trk_boxes[ti], det_boxes)

            # Greedy match on IoU
            while True:
                flat_idx = np.argmax(iou_mat)
                ti, di = np.unravel_index(flat_idx, iou_mat.shape)
                max_iou = iou_mat[ti, di]
                if max_iou < self.iou_threshold:
                    break
                g_trk = trk_idx[ti]
                g_det = det_idx[di]
                if (g_trk in matched_trk) or (g_det in matched_det):
                    iou_mat[ti, di] = -1.0
                    continue
                # assign
                assignments.append((g_trk, g_det))
                matched_trk.add(g_trk)
                matched_det.add(g_det)
                # invalidate row and col
                iou_mat[ti, :] = -1.0
                iou_mat[:, di] = -1.0

        # Update matched tracks
        for trk_i, det_i in assignments:
            self.tracks[trk_i].update(boxes_in[det_i], scores_in[det_i])

        # Create new tracks for unmatched detections
        for di in range(len(boxes_in)):
            if di not in matched_det:
                self._add_track(boxes_in[di], scores_in[di], int(cls_in[di]))

        # Prune stale tracks
        self._prune()

        # Output only tracks updated this frame and reaching min_hits
        out_boxes = []
        out_scores = []
        out_cls = []
        out_ids = []
        for t in self.tracks:
            if t.time_since_update == 0 and t.hits >= self.min_hits:
                out_boxes.append(t.bbox)
                out_scores.append(t.score)
                out_cls.append(t.cls_id)
                out_ids.append(t.id)

        if len(out_boxes) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )

        return (
            np.stack(out_boxes, axis=0).astype(np.float32),
            np.array(out_scores, dtype=np.float32),
            np.array(out_cls, dtype=np.int32),
            np.array(out_ids, dtype=np.int32),
        )
