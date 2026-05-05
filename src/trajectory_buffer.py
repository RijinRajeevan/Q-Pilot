import collections
import numpy as np

class TrajectoryBuffer:
    def __init__(self, time_steps=5, fps_estimate=30):
        self.time_steps = time_steps
        self.dt = 1.0 / fps_estimate
        # obj_id -> deque of features
        self.buffers = collections.defaultdict(lambda: collections.deque(maxlen=self.time_steps))
        
    def update(self, active_tracks, ego_resolution=(640, 360)):
        """
        Updates the buffer given the new frame's SORT tracks.
        active_tracks: format [[x1, y1, x2, y2, id], ...]
        ego_resolution: Base reference tuple to compute ego-relative positions.
        """
        cx_ego, cy_ego = ego_resolution[0] / 2.0, ego_resolution[1] # Ego is bottom center
        current_ids = set()
        
        for track in active_tracks:
            x1, y1, x2, y2, obj_id = track
            obj_id = int(obj_id)
            current_ids.add(obj_id)
            
            # Compute center of object
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            
            # Center relativity to Ego
            rel_x = cx - cx_ego
            rel_y = cy_ego - cy  # Y grows upwards from ego's perspective
            
            # Normalize spatial layout to ~[0, 1] realm assuming 640x360
            norm_x = rel_x / (ego_resolution[0] / 2.0)
            norm_y = rel_y / ego_resolution[1]
            
            # Compute derivatives if we have history
            prev = self.buffers[obj_id][-1] if len(self.buffers[obj_id]) > 0 else None
            if prev is not None:
                vel_x = (norm_x - prev['nx']) / self.dt
                vel_y = (norm_y - prev['ny']) / self.dt
                
                prev_vx, prev_vy = prev['vx'], prev['vy']
                acc_x = (vel_x - prev_vx) / self.dt
                acc_y = (vel_y - prev_vy) / self.dt
            else:
                vel_x, vel_y, acc_x, acc_y = 0.0, 0.0, 0.0, 0.0
                
            state = {
                'px': cx, 'py': cy,      # True Pixel coordinates
                'nx': norm_x, 'ny': norm_y, # Normalized Coordinates
                'vx': vel_x, 'vy': vel_y,   # Normalized Velocity
                'ax': acc_x, 'ay': acc_y    # Normalized Acceleration
            }
            
            self.buffers[obj_id].append(state)
            
        # Optional: Prune dead tracks
        dead_keys = set(self.buffers.keys()) - current_ids
        for k in dead_keys:
            del self.buffers[k]
            
    def get_ready_tensors(self):
        """
        Returns active objects with full T=5 sequential history ready for PyTorch batched inference.
        Returns: 
           ids: [list of obj_ids]
           batch: [batch_size, T=5, features=6]  (assume [nx, ny, vx, vy, ax, ay])
           pixels: [list of current (px, py) for drawing overlays]
        """
        ready_ids = []
        batch = []
        pixels = []
        
        for obj_id, history in self.buffers.items():
            if len(history) == self.time_steps:
                ready_ids.append(obj_id)
                pixels.append((history[-1]['px'], history[-1]['py']))
                seq = []
                for point in history:
                    seq.append([point['nx'], point['ny'], point['vx'], point['vy'], point['ax'], point['ay']])
                batch.append(seq)
                
        if len(batch) == 0:
            return [], None, []
            
        # Return as normalized numpy array ready for torch.from_numpy()
        return ready_ids, np.array(batch, dtype=np.float32), pixels
