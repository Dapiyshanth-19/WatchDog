"""
SORT – Simple Online and Realtime Tracking
Compact self-contained implementation using filterpy + scipy.
No external 'sort' package required.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


def _iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_test = (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1])
    area_gt   = (bb_gt[2]-bb_gt[0])   * (bb_gt[3]-bb_gt[1])
    union = area_test + area_gt - inter
    return inter / union if union > 0 else 0.0


def _iou_matrix(dets: np.ndarray, trks: np.ndarray) -> np.ndarray:
    mat = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            mat[d, t] = _iou(det, trk)
    return mat


def _linear_assignment(cost_matrix: np.ndarray):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return np.stack([row_ind, col_ind], axis=1)


def _convert_bbox_to_z(bbox):
    """[x1,y1,x2,y2] → [cx, cy, s, r] where s=area, r=aspect ratio."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([[x], [y], [s], [r]])


def _convert_x_to_bbox(x, score=None):
    """[cx, cy, s, r] → [x1, y1, x2, y2]."""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w if w > 0 else 0
    box = [x[0]-w/2, x[1]-h/2, x[0]+w/2, x[1]+h/2]
    if score is None:
        return np.array(box).reshape(1, 4)
    return np.array([*box, score]).reshape(1, 5)


class _KalmanBoxTracker:
    """Maintains the state of a single tracked object with a Kalman filter."""
    count = 0

    def __init__(self, bbox):
        _KalmanBoxTracker.count += 1
        self.id = _KalmanBoxTracker.count
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # State: [cx, cy, s, r, vx, vy, vs]
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = _convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.hit_streak = 0
        self.history = []

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hit_streak += 1
        self.kf.update(_convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(_convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return _convert_x_to_bbox(self.kf.x)


class Sort:
    """
    SORT tracker.

    Usage::

        tracker = Sort(max_age=10, min_hits=2, iou_threshold=0.25)
        # Each frame:
        tracks = tracker.update(dets)   # dets: np.ndarray shape (N,5) [x1,y1,x2,y2,score]
        # tracks: np.ndarray shape (M,5) [x1,y1,x2,y2,track_id]
    """

    def __init__(self, max_age=10, min_hits=2, iou_threshold=0.25):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[_KalmanBoxTracker] = []
        self.frame_count = 0
        _KalmanBoxTracker.count = 0  # reset IDs on new Sort instance

    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dets : np.ndarray, shape (N, 5) — [x1, y1, x2, y2, score]
               Pass empty array shape (0, 5) if no detections.

        Returns
        -------
        np.ndarray, shape (M, 5) — [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        # Predict new locations of existing tracks
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = [*pos, 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        matched, unmatched_dets, unmatched_trks = self._associate(dets, trks)

        # Update matched trackers
        for d, t in matched:
            self.trackers[t].update(dets[d, :4])

        # Create new trackers for unmatched detections
        for d in unmatched_dets:
            self.trackers.append(_KalmanBoxTracker(dets[d, :4]))

        # Build output, prune dead tracks
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append([*d, trk.id])
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return np.array(ret) if ret else np.empty((0, 5))

    def _associate(self, dets, trks):
        if len(trks) == 0:
            return [], list(range(len(dets))), []

        iou_mat = _iou_matrix(dets[:, :4], trks[:, :4])
        cost = 1.0 - iou_mat
        indices = _linear_assignment(cost)

        unmatched_dets = [d for d in range(len(dets)) if d not in indices[:, 0]]
        unmatched_trks = [t for t in range(len(trks)) if t not in indices[:, 1]]

        matched = []
        for d, t in indices:
            if iou_mat[d, t] < self.iou_threshold:
                unmatched_dets.append(d)
                unmatched_trks.append(t)
            else:
                matched.append((d, t))

        return matched, unmatched_dets, unmatched_trks
