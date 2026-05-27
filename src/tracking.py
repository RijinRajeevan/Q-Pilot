"""
Q-Pilot V7 — ByteTrack-style Multi-Object Tracker
Kalman Filter prediction + IoU-based assignment with byte-level association.
Handles high-confidence and low-confidence detections separately for robustness.
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test, bb_gt):
    """Vectorised IoU between two sets of bounding boxes [x1,y1,x2,y2]."""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    union = area_test + area_gt - inter
    return inter / np.maximum(union, 1e-6)


class KalmanBoxTracker:
    """Kalman filter tracker for a single bounding box."""
    _id_counter = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # State: [cx, cy, area, aspect_ratio, vx, vy, v_area]
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self._bbox_to_z(bbox)

        self.time_since_update = 0
        KalmanBoxTracker._id_counter += 1
        self.id = KalmanBoxTracker._id_counter
        self.hits = 1
        self.hit_streak = 1
        self.age = 1

    @classmethod
    def reset_counter(cls):
        """Reset the ID counter — call when switching scenarios."""
        cls._id_counter = 0

    @staticmethod
    def _bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.
        cy = bbox[1] + h / 2.
        return np.array([cx, cy, w * h, w / max(h, 1e-6)]).reshape((4, 1))

    def _z_to_bbox(self, z):
        w = np.sqrt(max(z[2] * z[3], 1e-6))
        h = max(z[2] / max(w, 1e-6), 1e-6)
        return np.array([z[0] - w / 2., z[1] - h / 2., z[0] + w / 2., z[1] + h / 2.]).flatten()

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def get_state(self):
        return self._z_to_bbox(self.kf.x.flatten())


class ByteTracker:
    """
    ByteTrack-style tracker:
    1. First association pass with high-confidence detections (>= high_thresh)
    2. Second association pass with low-confidence detections (>= low_thresh)
    3. Unmatched high-conf detections become new tracks
    """

    def __init__(self, max_age=12, min_hits=3, iou_threshold=0.3,
                 high_thresh=0.5, low_thresh=0.1):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Args:
            dets: (N, 5) array of [x1, y1, x2, y2, confidence]
        Returns:
            (M, 5) array of [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        # Predict new positions for all existing trackers
        predicted = []
        to_del = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            if np.any(np.isnan(pred)):
                to_del.append(i)
            else:
                predicted.append(pred)
        for i in reversed(to_del):
            self.trackers.pop(i)

        trk_bboxes = np.array(predicted) if predicted else np.empty((0, 4))

        # Split detections into high and low confidence
        if len(dets) > 0:
            high_mask = dets[:, 4] >= self.high_thresh
            low_mask = (dets[:, 4] >= self.low_thresh) & (dets[:, 4] < self.high_thresh)
            dets_high = dets[high_mask]
            dets_low = dets[low_mask]
        else:
            dets_high = np.empty((0, 5))
            dets_low = np.empty((0, 5))

        # ── First association: high-confidence dets vs all tracks ──
        unmatched_trk_indices = list(range(len(self.trackers)))
        if len(dets_high) > 0 and len(trk_bboxes) > 0:
            matched, unmatched_det_high, unmatched_trks = self._associate(
                dets_high[:, :4], trk_bboxes, self.iou_threshold
            )
            for d_idx, t_idx in matched:
                self.trackers[t_idx].update(dets_high[d_idx, :4])
            unmatched_trk_indices = list(unmatched_trks)
        else:
            unmatched_det_high = list(range(len(dets_high)))

        # ── Second association: low-confidence dets vs remaining tracks ──
        if len(dets_low) > 0 and len(unmatched_trk_indices) > 0:
            remaining_trk_bboxes = np.array([
                self.trackers[i].get_state() for i in unmatched_trk_indices
            ])
            matched_low, _, _ = self._associate(
                dets_low[:, :4], remaining_trk_bboxes, 0.5  # stricter for low-conf
            )
            for d_idx, t_local_idx in matched_low:
                real_t_idx = unmatched_trk_indices[t_local_idx]
                self.trackers[real_t_idx].update(dets_low[d_idx, :4])
                unmatched_trk_indices.remove(real_t_idx)

        # ── Create new tracks from unmatched high-confidence detections ──
        for i in unmatched_det_high:
            self.trackers.append(KalmanBoxTracker(dets_high[i, :4]))

        # ── Build output ──
        ret = []
        for trk in reversed(self.trackers):
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)
                continue
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                state = trk.get_state()
                ret.append(np.append(state, trk.id))

        return np.array(ret).reshape(-1, 5) if ret else np.empty((0, 5))

    def _associate(self, dets, trks, iou_thresh):
        """Hungarian algorithm association."""
        if len(trks) == 0:
            return [], list(range(len(dets))), []
        if len(dets) == 0:
            return [], [], list(range(len(trks)))

        iou_matrix = iou_batch(dets, trks)
        matched_indices = []

        if min(iou_matrix.shape) > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= iou_thresh:
                    matched_indices.append([r, c])

        matched_det = set(m[0] for m in matched_indices)
        matched_trk = set(m[1] for m in matched_indices)
        unmatched_dets = [i for i in range(len(dets)) if i not in matched_det]
        unmatched_trks = [i for i in range(len(trks)) if i not in matched_trk]

        return matched_indices, unmatched_dets, unmatched_trks
