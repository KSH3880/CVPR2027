import torch
import numpy as np

class SingleTrajManager():
    def __init__(self, traj_data, device, dt, speed_min, speed_max, accel_max, extend, extend_dist):

        self._device = device
        self._dt = dt
        self._speed_min = speed_min
        self._speed_max = speed_max
        self._accel_max = accel_max
        
        self._end_points = traj_data

        if len(self._end_points.shape) == 2:
            self._end_points = np.hstack((self._end_points, np.zeros((self._end_points.shape[0], 1))))
        else:
            self._end_points[..., -1] = 0 # clean z-axis

        # extend the trajector
        if extend:
            vec = self._end_points[-1] - self._end_points[-2]
            vec_norm = np.linalg.norm(vec, ord=2)
            vec /= vec_norm

            extra_end_point = self._end_points[-1] + extend_dist * vec
            self._end_points = np.vstack((self._end_points, extra_end_point.reshape(1, 3)))
        
        self._num_points = self._end_points.shape[0]

        self._segm_len = [] # length of each segm
        self._segm_dir = [] # direction of each segm
        for i in range(self._num_points - 1):
            vec = self._end_points[i + 1] - self._end_points[i]
            vec_norm = np.linalg.norm(vec, ord=2)
            vec /= vec_norm
            self._segm_len.append(vec_norm)
            self._segm_dir.append(vec)
        self._segm_len = np.array(self._segm_len)
        self._segm_dir = np.array(self._segm_dir)
        self._num_segm = len(self._segm_len)

        self._verts = []
        self._verts.append(self._end_points[0])

        base_speed = (self._speed_max - self._speed_min) * np.random.rand(1) + self._speed_min
        counter = 0
        while True:
            dspeed = 2 * np.random.rand(1) - 1
            dspeed *= self._accel_max * self._dt

            curr_speed = base_speed + dspeed
            curr_speed = np.clip(curr_speed, self._speed_min, self._speed_max) # clip speed
            base_speed = curr_speed # update

            move_length = curr_speed * self._dt
            start_vert = self._verts[-1]
            # search a valid point on the path
            while counter < self._num_segm:

                new_vert = start_vert + self._segm_dir[counter] * move_length

                # check whether valid
                max_move_length_this_segm = np.linalg.norm(self._end_points[counter + 1] - start_vert, ord=2)
                if move_length > max_move_length_this_segm:
                    move_length -= max_move_length_this_segm
                    start_vert = self._end_points[counter + 1]
                    counter += 1
                else:
                    break

            if counter < self._num_segm:
                self._verts.append(new_vert)
            else:
                self._verts.append(self._end_points[-1])
                assert counter == self._num_points - 1
                break
        
        self._verts = np.array(self._verts)
        self._num_verts = len(self._verts)
        self._episode_dur = (self._num_verts - 1) * self._dt

        # to torch
        self._verts = torch.tensor(self._verts, dtype=torch.float32, device=self._device)

    def calc_pos(self, times):
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self._verts[seg_id0]
        pos1 = self._verts[seg_id1]

        lerp = lerp.unsqueeze(-1)
        pos = (1.0 - lerp) * pos0 + lerp * pos1

        return pos

    def get_num_verts(self):
        return self._verts.shape[0]

    def get_num_segs(self):
        return self.get_num_verts() - 1

    def get_traj_duration(self):
        dur = self.get_num_segs() * self._dt
        return dur

class CustomTrajGenerator():
    def __init__(self, trajs, device, dt, speed_min, speed_max, accel_max, extend, extend_dist):

        self._device = device

        # load data
        self._traj_seqs = [np.array(v, dtype=np.float32) for k, v in trajs.items()]
        self._traj_managers = []
        for seq in self._traj_seqs:
            self._traj_managers.append(SingleTrajManager(seq, device, dt, speed_min, speed_max, accel_max, extend, extend_dist))
        self._num_trajs = len(self._traj_managers)

        return

    def calc_pos(self, traj_ids, times):
        n = len(traj_ids)
        pos = torch.zeros((n, 3), dtype=torch.float32, device=self._device)
        
        for i in range(self._num_trajs):
            mask = (traj_ids == i)
            pos[mask] = self._traj_managers[i].calc_pos(times[mask])
        
        return pos

    def get_traj_verts(self, traj_id): 
        return self._traj_managers[traj_id]._verts
