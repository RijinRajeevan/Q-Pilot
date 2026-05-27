"""
Q-Pilot V7 — Trajectory History Buffer
Maintains per-object sliding window of kinematic states for sequence-based prediction.
20-frame history, computes velocity, acceleration, heading, delta positions.
"""
import collections
import numpy as np
import math


class TrajectoryBuffer:
    """
    Per-object trajectory history buffer.
    Stores 20-frame sliding windows of kinematic features per tracked object.
    Provides ready-to-use input sequences for GRU/LSTM/QNN inference.
    """

    def __init__(self, history_len=20, predict_len=5, fps_estimate=20):
        self.history_len = history_len
        self.predict_len = predict_len
        self.dt = 1.0 / fps_estimate

        # obj_id -> deque of feature dicts
        self.buffers: dict[int, collections.deque] = {}

        # obj_id -> class_name
        self.class_map: dict[int, str] = {}

    def update(self, tracks, class_map: dict, frame_w=640, frame_h=360):
        """
        Update buffer with new frame's tracked objects.

        Args:
            tracks: (N, 5) array of [x1, y1, x2, y2, track_id]
            class_map: dict mapping track_id -> class_name
            frame_w, frame_h: frame resolution for normalization
        """
        ego_cx = frame_w / 2.0
        ego_cy = frame_h  # Ego is at bottom-center

        current_ids = set()

        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            tid = int(tid)
            current_ids.add(tid)

            # Store class
            if tid in class_map:
                self.class_map[tid] = class_map[tid]

            # Compute center
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1

            # Relative to ego (normalized)
            rel_x = (cx - ego_cx) / (frame_w / 2.0)  # [-1, 1]
            rel_y = (ego_cy - cy) / frame_h           # [0, 1] (higher = further away)

            # Initialize buffer if new track
            if tid not in self.buffers:
                self.buffers[tid] = collections.deque(maxlen=self.history_len)

            prev = self.buffers[tid][-1] if len(self.buffers[tid]) > 0 else None

            # Compute velocity (normalized units/frame)
            if prev is not None:
                dx = rel_x - prev['rel_x']
                dy = rel_y - prev['rel_y']
                # EMA smoothing
                alpha = 0.4
                vx = alpha * (dx / self.dt) + (1 - alpha) * prev['vx']
                vy = alpha * (dy / self.dt) + (1 - alpha) * prev['vy']
                # Acceleration
                ax = alpha * ((vx - prev['vx']) / self.dt) + (1 - alpha) * prev['ax']
                ay = alpha * ((vy - prev['vy']) / self.dt) + (1 - alpha) * prev['ay']
                # Heading angle
                heading = math.atan2(dy, dx) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else prev['heading']
            else:
                vx, vy = 0.0, 0.0
                ax, ay = 0.0, 0.0
                heading = 0.0

            # Speed magnitude
            speed = math.sqrt(vx ** 2 + vy ** 2)

            # Distance from ego
            dist_from_ego = math.sqrt(rel_x ** 2 + rel_y ** 2)

            # Lane offset (approximate: how far laterally from center)
            lane_offset = rel_x

            state = {
                # Pixel coordinates (for overlay rendering)
                'px': float(cx),
                'py': float(cy),
                'bw': float(bw),
                'bh': float(bh),
                # Normalized coordinates
                'rel_x': float(rel_x),
                'rel_y': float(rel_y),
                # Kinematics (normalized)
                'vx': float(vx),
                'vy': float(vy),
                'ax': float(ax),
                'ay': float(ay),
                'speed': float(speed),
                'heading': float(heading),
                # Spatial features
                'lane_offset': float(lane_offset),
                'dist_from_ego': float(dist_from_ego),
            }

            self.buffers[tid].append(state)

        # Prune dead tracks (not seen for this frame)
        dead_keys = set(self.buffers.keys()) - current_ids
        for k in dead_keys:
            del self.buffers[k]
            if k in self.class_map:
                del self.class_map[k]

    def get_sequence_inputs(self, min_history=10):
        """
        Get input sequences for trajectory prediction models.
        Only returns tracks with at least `min_history` frames of data.

        Returns:
            ids: list of track IDs
            sequences: np.array of shape (N, min_history, n_features)
            pixel_positions: list of current (px, py) for overlay
            kinematics: list of dicts with current vel/accel/heading per object
        """
        ids = []
        sequences = []
        pixels = []
        kinematics = []

        # Feature order: rel_x, rel_y, vx, vy, ax, ay, heading, lane_offset, speed, dist_from_ego
        feature_keys = ['rel_x', 'rel_y', 'vx', 'vy', 'ax', 'ay',
                        'heading', 'lane_offset', 'speed', 'dist_from_ego']

        for tid, history in self.buffers.items():
            if len(history) >= min_history:
                ids.append(tid)
                pixels.append((history[-1]['px'], history[-1]['py']))

                # Extract last min_history frames as feature matrix
                recent = list(history)[-min_history:]
                seq = []
                for frame_state in recent:
                    feat_vec = [frame_state[k] for k in feature_keys]
                    seq.append(feat_vec)
                sequences.append(seq)

                # Current kinematics for display
                last = history[-1]
                kinematics.append({
                    'vx': last['vx'],
                    'vy': last['vy'],
                    'ax': last['ax'],
                    'ay': last['ay'],
                    'speed': last['speed'],
                    'heading': last['heading'],
                    'dist_from_ego': last['dist_from_ego'],
                    'lane_offset': last['lane_offset'],
                })

        if not sequences:
            return [], None, [], []

        return ids, np.array(sequences, dtype=np.float32), pixels, kinematics

    def get_all_tracks_info(self):
        """Get basic info for all currently buffered tracks."""
        info = {}
        for tid, history in self.buffers.items():
            if len(history) > 0:
                last = history[-1]
                info[tid] = {
                    'px': last['px'],
                    'py': last['py'],
                    'bw': last['bw'],
                    'bh': last['bh'],
                    'vx': last['vx'],
                    'vy': last['vy'],
                    'speed': last['speed'],
                    'heading': last['heading'],
                    'history_len': len(history),
                    'class': self.class_map.get(tid, 'car'),
                }
        return info
