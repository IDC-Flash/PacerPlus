# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import joblib
import random
from amp.utils.flags import flags
from amp.env.tasks.base_task import PORT, SERVER

class TrajGenerator():
    def __init__(self, num_envs, episode_dur, num_verts, device, dtheta_max,
                 speed_min, speed_max, accel_max, sharp_turn_prob):


        self._device = device
        self._dt = episode_dur / (num_verts - 1)
        self._dtheta_max = dtheta_max
        self._speed_min = speed_min
        self._speed_max = speed_max
        self._accel_max = accel_max
        self._sharp_turn_prob = sharp_turn_prob

        self._verts_flat = torch.zeros((num_envs * num_verts, 3), dtype=torch.float32, device=self._device)
        self._verts = self._verts_flat.view((num_envs, num_verts, 3))

        env_ids = torch.arange(self.get_num_envs(), dtype=np.int)

        # self.traj_data = joblib.load("data/traj/traj_data.pkl")
        self.heading = torch.zeros(num_envs, 1)
        return

    def reset(self, env_ids, init_pos):
        n = len(env_ids)
        if (n > 0):
            num_verts = self.get_num_verts()
            dtheta = 2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0 # Sample the angles at each waypoint
            dtheta *= self._dtheta_max * self._dt

            dtheta_sharp = np.pi * (2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0) # Sharp Angles Angle
            sharp_probs = self._sharp_turn_prob * torch.ones_like(dtheta)
            sharp_mask = torch.bernoulli(sharp_probs) == 1.0
            dtheta[sharp_mask] = dtheta_sharp[sharp_mask]

            dtheta[:, 0] = np.pi * (2 * torch.rand([n], device=self._device) - 1.0) # Heading


            dspeed = 2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0
            dspeed *= self._accel_max * self._dt
            dspeed[:, 0] = (self._speed_max - self._speed_min) * torch.rand([n], device=self._device) + self._speed_min # Speed

            speed = torch.zeros_like(dspeed)
            speed[:, 0] = dspeed[:, 0]
            for i in range(1, dspeed.shape[-1]):
                speed[:, i] = torch.clip(speed[:, i - 1] + dspeed[:, i], self._speed_min, self._speed_max)

            ################################################
            if flags.fixed_path:
                dtheta[:, :] = 0 # ZL: Hacking to make everything 0
                dtheta[0, 0] = 0 # ZL: Hacking to create collision
                if len(dtheta) > 1:
                    dtheta[1, 0] = -np.pi # ZL: Hacking to create collision
                speed[:] = (self._speed_min + self._speed_max)/2
            ################################################

            if flags.slow:
                speed[:] = speed/4


            dtheta = torch.cumsum(dtheta, dim=-1)

            seg_len = speed * self._dt

            dpos = torch.stack([torch.cos(dtheta), -torch.sin(dtheta), torch.zeros_like(dtheta)], dim=-1)
            dpos *= seg_len.unsqueeze(-1)
            dpos[..., 0, 0:2] += init_pos[..., 0:2]
            vert_pos = torch.cumsum(dpos, dim=-2)

            self._verts[env_ids, 0, 0:2] = init_pos[..., 0:2]
            self._verts[env_ids, 1:] = vert_pos

            ####### ZL: Loading random real-world trajectories #######
            if flags.real_path:
                rids = random.sample(self.traj_data.keys(), n)
                traj = torch.stack([
                    torch.from_numpy(
                        self.traj_data[id]['coord_dense'])[:num_verts]
                    for id in rids
                ],
                                   dim=0).to(self._device).float()

                traj[..., 0:2] = traj[..., 0:2] - (traj[..., 0, 0:2] - init_pos[..., 0:2])[:, None]
                self._verts[env_ids] = traj

        return

    def input_new_trajs(self, env_ids):
        import json
        import requests
        from scipy.interpolate import interp1d
        x = requests.get(
            f'http://{SERVER}:{PORT}/path?num_envs={len(env_ids)}')
        print(SERVER)
        data_lists = [value for idx, value in x.json().items()]
        coord = np.array(data_lists)
        x = np.linspace(0, coord.shape[1] - 1, num = coord.shape[1])
        fx = interp1d(x, coord[..., 0], kind='linear')
        fy = interp1d(x, coord[..., 1], kind='linear')
        x4 = np.linspace(0, coord.shape[1] - 1, num = coord.shape[1] * 10)
        coord_dense = np.stack([fx(x4), fy(x4), np.zeros([len(env_ids), x4.shape[0]])], axis = -1)
        coord_dense = np.concatenate([coord_dense, coord_dense[..., -1:, :]], axis = -2)
        self._verts[env_ids] = torch.from_numpy(coord_dense).float().to(env_ids.device)
        return self._verts[env_ids]


    def get_num_verts(self):
        return self._verts.shape[1]

    def get_num_segs(self):
        return self.get_num_verts() - 1

    def get_num_envs(self):
        return self._verts.shape[0]

    def get_traj_duration(self):
        num_verts = self.get_num_verts()
        dur = num_verts * self._dt
        return  dur

    def get_traj_verts(self, traj_id):
        return self._verts[traj_id]

    def calc_pos(self, traj_ids, times):
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self._verts_flat[traj_ids * num_verts + seg_id0]
        pos1 = self._verts_flat[traj_ids * num_verts + seg_id1]

        lerp = lerp.unsqueeze(-1)
        pos = (1.0 - lerp) * pos0 + lerp * pos1

        return pos

    def mock_calc_pos(self, env_ids, traj_ids, times, query_value_gradient):
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self._verts_flat[traj_ids * num_verts + seg_id0]
        pos1 = self._verts_flat[traj_ids * num_verts + seg_id1]

        lerp = lerp.unsqueeze(-1)
        pos = (1.0 - lerp) * pos0 + lerp * pos1

        new_obs, func = query_value_gradient(env_ids, pos)
        if not new_obs is None:
            # ZL: computes grad
            with torch.enable_grad():
                new_obs.requires_grad_(True)
                new_val = func(new_obs)

                disc_grad = torch.autograd.grad(
                    new_val,
                    new_obs,
                    grad_outputs=torch.ones_like(new_val),
                    create_graph=False,
                    retain_graph=True,
                    only_inputs=True)

        return pos
    
    def get_velocity(self, traj_ids, times):
        pos_total = self.calc_pos(traj_ids.flatten(), times.flatten())
        pos_total = pos_total.reshape(traj_ids.shape[0], times.shape[1], pos_total.shape[1])
        pos = pos_total[:, 0]
        pos_next = pos_total[:, -1]
        vel = (pos_next - pos) / (times[:, -1] - times[:, 0]).unsqueeze(-1)
        return vel



class HybirdTrajGenerator():
    def __init__(self, num_envs, episode_dur, num_verts, device, dtheta_max,
                 speed_min, speed_max, accel_max, sharp_turn_prob, traj_sample_timestep, max_episode_steps):


        self._device = device
        self._dt = episode_dur / (num_verts - 1)
        self._dtheta_max = dtheta_max
        self._speed_min = speed_min
        self._speed_max = speed_max
        self._accel_max = accel_max
        self._sharp_turn_prob = sharp_turn_prob

        self._verts_flat = torch.zeros((num_envs * num_verts, 3), dtype=torch.float32, device=self._device)
        self._verts = self._verts_flat.view((num_envs, num_verts, 3))

        env_ids = torch.arange(self.get_num_envs(), dtype=np.int)

        # self.traj_data = joblib.load("data/traj/traj_data.pkl")
        self.heading = torch.zeros(num_envs, 1)
        self.max_episode_steps = max_episode_steps
        self.traj_position = torch.zeros((num_envs, max_episode_steps, 3), dtype=torch.float32, device=self._device)
        self.diff_xy = torch.zeros((num_envs, 2), dtype=torch.float32, device=self._device)
        return

    def reset(self, env_ids, init_pos, real_traj, real_traj_index):
        n = len(env_ids)
        if (n > 0):
            num_verts = self.get_num_verts()
            dtheta = 2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0 # Sample the angles at each waypoint
            dtheta *= self._dtheta_max * self._dt

            dtheta_sharp = np.pi * (2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0) # Sharp Angles Angle
            sharp_probs = self._sharp_turn_prob * torch.ones_like(dtheta)
            sharp_mask = torch.bernoulli(sharp_probs) == 1.0
            dtheta[sharp_mask] = dtheta_sharp[sharp_mask]

            dtheta[:, 0] = np.pi * (2 * torch.rand([n], device=self._device) - 1.0) # Heading


            dspeed = 2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0
            dspeed *= self._accel_max * self._dt
            dspeed[:, 0] = (self._speed_max - self._speed_min) * torch.rand([n], device=self._device) + self._speed_min # Speed

            speed = torch.zeros_like(dspeed)
            speed[:, 0] = dspeed[:, 0]
            for i in range(1, dspeed.shape[-1]):
                speed[:, i] = torch.clip(speed[:, i - 1] + dspeed[:, i], self._speed_min, self._speed_max)

            ################################################
            if flags.fixed_path:
                dtheta[:, :] = 0 # ZL: Hacking to make everything 0
                dtheta[0, 0] = 0 # ZL: Hacking to create collision
                if len(dtheta) > 1:
                    dtheta[1, 0] = -np.pi # ZL: Hacking to create collision
                speed[:] = (self._speed_min + self._speed_max)/2
            ################################################

            if flags.slow:
                speed[:] = speed/4


            dtheta = torch.cumsum(dtheta, dim=-1)

            seg_len = speed * self._dt

            dpos = torch.stack([torch.cos(dtheta), -torch.sin(dtheta), torch.zeros_like(dtheta)], dim=-1)
            dpos *= seg_len.unsqueeze(-1)
            dpos[..., 0, 0:2] += init_pos[..., 0:2]
            vert_pos = torch.cumsum(dpos, dim=-2)

            self._verts[env_ids, 0, 0:2] = init_pos[..., 0:2]
            self._verts[env_ids, 1:] = vert_pos

            ########### assign real traj ############
            dt_fps = 1 / 30
            timestep_beg = torch.zeros((env_ids.shape[0])).to(self._device).int() * dt_fps
            timesteps = torch.arange(self.max_episode_steps, device=self._device, dtype=torch.float)
            timesteps = timesteps * dt_fps
            traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps
            env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)
            traj_samples_flat = self._calc_pos(env_ids_tiled.flatten(), traj_timesteps.flatten())
            traj_samples = traj_samples_flat.reshape(env_ids.shape[0], self.max_episode_steps, 3)
            self.traj_position[env_ids] = traj_samples
            self.real_traj = real_traj
            self.real_traj_index = real_traj_index

            for i in range(n):
                assign_position = self.traj_position[env_ids[i], real_traj_index[env_ids[i]], 0:2]
                synthetic_traj_diff = assign_position - real_traj[env_ids[i]][0:1, 0:2]
                real_traj[env_ids[i]] = real_traj[env_ids[i]][:, 0:2] - real_traj[env_ids[i]][0:1, 0:2] + assign_position.unsqueeze(0)                # synthetic_traj_diff = self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]-1, 0:2] - end_real_traj
                real_traj_direction = (real_traj[env_ids[i]][-1, 0:2] - real_traj[env_ids[i]][0, 0:2]) / real_traj[env_ids[i]].shape[0]
                real_traj_direction = real_traj_direction.unsqueeze(0)
                real_traj_direction = real_traj_direction / torch.norm(real_traj_direction, dim=1, keepdim=True)
                real_traj_rot = torch.cat([real_traj_direction, torch.stack([real_traj_direction[:, 1], real_traj_direction[:, 0]], dim=1)], dim=0) #### invserse rotation matrix of real direction

                self.diff_xy[env_ids[i]] = synthetic_traj_diff[0]

                self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]:, 0:2] -= \
                    self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]-1, 0:2].unsqueeze(0)

                synthetic_traj_direction = self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0], 0:2]   
                synthetic_traj_direction = synthetic_traj_direction.unsqueeze(0)
                synthetic_traj_direction = synthetic_traj_direction / torch.norm(synthetic_traj_direction, dim=1, keepdim=True)
                synthetic_traj_rot = torch.cat([synthetic_traj_direction, torch.stack([-synthetic_traj_direction[:, 1], synthetic_traj_direction[:, 0]], dim=1)], dim=0) #### invserse rotation matrix of synthetic direction


                self.traj_position[env_ids[i], real_traj_index[env_ids[i]]:real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0], 0:2] = real_traj[env_ids[i]]
                if not torch.isnan(real_traj_rot).any() and not torch.isnan(synthetic_traj_rot).any():
                    self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]:, 0:2] = \
                        torch.matmul(torch.matmul(self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]:, 0:2], torch.inverse(synthetic_traj_rot)), real_traj_rot)
                self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]:, 0:2] += \
                    self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]-1, 0:2].unsqueeze(0)

                ####### compute the orientation  change ##########

        self.traj_position_flat = self.traj_position.reshape(-1, 3)
        return
    

    def get_num_verts(self):
        return self._verts.shape[1]

    def get_num_segs(self):
        return self.get_num_verts() - 1

    def get_num_envs(self):
        return self._verts.shape[0]

    def get_traj_duration(self):
        num_verts = self.get_num_verts()
        dur = num_verts * self._dt
        return  dur

    def get_traj_verts(self, traj_id):
        return self._verts[traj_id]

    def _calc_pos(self, traj_ids, times):
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self._verts_flat[traj_ids * num_verts + seg_id0]
        pos1 = self._verts_flat[traj_ids * num_verts + seg_id1]

        lerp = lerp.unsqueeze(-1)
        pos = (1.0 - lerp) * pos0 + lerp * pos1

        return pos


    def calc_pos(self, traj_ids, times):
        traj_dur = self.get_traj_duration()
        num_verts = self.max_episode_steps 
        num_segs = num_verts - 1

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self.traj_position_flat[traj_ids * num_verts + seg_id0]
        pos1 = self.traj_position_flat[traj_ids * num_verts + seg_id1]

        lerp = lerp.unsqueeze(-1)
        pos = (1.0 - lerp) * pos0 + lerp * pos1
        return pos
    
    def get_diff_xy(self, traj_ids):
        return self.diff_xy[traj_ids]
    

    def get_velocity(self, traj_ids, times):
        pos_total = self.calc_pos(traj_ids.flatten(), times.flatten())
        pos_total = pos_total.reshape(traj_ids.shape[0], times.shape[1], pos_total.shape[1])
        pos = pos_total[:, 0]
        pos_next = pos_total[:, -1]
        vel = (pos_next - pos) / (times[:, -1] - times[:, 0]).unsqueeze(-1)
        return vel

    def input_new_trajs(self, env_ids):
        import json
        import requests
        from scipy.interpolate import interp1d
        x = requests.get(
            f'http://{SERVER}:{PORT}/path?num_envs={len(env_ids)}')
        print(SERVER)
        data_lists = [value for idx, value in x.json().items()]
        coord = np.array(data_lists)
        x = np.linspace(0, coord.shape[1] - 1, num = coord.shape[1])
        fx = interp1d(x, coord[..., 0], kind='linear')
        fy = interp1d(x, coord[..., 1], kind='linear')
        x4 = np.linspace(0, coord.shape[1] - 1, num = coord.shape[1] * 10)
        coord_dense = np.stack([fx(x4), fy(x4), np.zeros([len(env_ids), x4.shape[0]])], axis = -1)
        coord_dense = np.concatenate([coord_dense, coord_dense[..., -1:, :]], axis = -2)
        self._verts[env_ids] = torch.from_numpy(coord_dense).float().to(env_ids.device)
        return self._verts[env_ids]

    def insert_real_trajectory(self, env_ids, real_traj, real_traj_index):
        ########### assign real traj ############
        dt_fps = 1 / 30
        timestep_beg = torch.zeros((env_ids.shape[0])).to(self._device).int() * dt_fps
        timesteps = torch.arange(self.max_episode_steps, device=self._device, dtype=torch.float)
        timesteps = timesteps * dt_fps
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps
        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)
        traj_samples_flat = self._calc_pos(env_ids_tiled.flatten(), traj_timesteps.flatten())
        traj_samples = traj_samples_flat.reshape(env_ids.shape[0], self.max_episode_steps, 3)
        self.traj_position[env_ids] = traj_samples
        self.real_traj = real_traj
        self.real_traj_index = real_traj_index
        end_position = self._verts[env_ids, -1, 0:2]
        n = len(env_ids)

        for i in range(n):
            assign_position = self.traj_position[env_ids[i], real_traj_index[env_ids[i]], 0:2]
            synthetic_traj_diff = assign_position - real_traj[env_ids[i]][0:1, 0:2]
            real_traj[env_ids[i]] = real_traj[env_ids[i]][:, 0:2] - real_traj[env_ids[i]][0:1, 0:2] + assign_position.unsqueeze(0)                # synthetic_traj_diff = self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]-1, 0:2] - end_real_traj
            real_traj_direction = (real_traj[env_ids[i]][-1, 0:2] - real_traj[env_ids[i]][0, 0:2]) / real_traj[env_ids[i]].shape[0]
            real_traj_direction = real_traj_direction.unsqueeze(0)
            real_traj_direction = real_traj_direction / torch.norm(real_traj_direction, dim=1, keepdim=True)
            real_traj_rot = torch.cat([real_traj_direction, torch.stack([-real_traj_direction[:, 1], real_traj_direction[:, 0]], dim=1)], dim=0) #### invserse rotation matrix of real direction

            self.diff_xy[env_ids[i]] = synthetic_traj_diff[0]

            self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]:, 0:2] -= \
                self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]-1, 0:2].unsqueeze(0)

            synthetic_traj_direction = self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0], 0:2]   
            synthetic_traj_direction = synthetic_traj_direction.unsqueeze(0)
            synthetic_traj_direction = synthetic_traj_direction / torch.norm(synthetic_traj_direction, dim=1, keepdim=True)
            synthetic_traj_rot = torch.cat([synthetic_traj_direction, torch.stack([-synthetic_traj_direction[:, 1], synthetic_traj_direction[:, 0]], dim=1)], dim=0) #### invserse rotation matrix of synthetic direction


            self.traj_position[env_ids[i], real_traj_index[env_ids[i]]:real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0], 0:2] = real_traj[env_ids[i]]
            if not torch.isnan(real_traj_rot).any() and not torch.isnan(synthetic_traj_rot).any():
                self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]:, 0:2] = \
                    torch.matmul(torch.matmul(self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]:, 0:2], torch.inverse(synthetic_traj_rot)), real_traj_rot)
            self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]:, 0:2] += \
                self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]-1, 0:2].unsqueeze(0)
            remain_dist = end_position[env_ids[i]] - self.traj_position[env_ids[i], -1, :2]
            remain_length = self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]:, 0:2].shape[0]
            per_frame_dist = remain_dist / remain_length
            offset = torch.arange(1, remain_length+1, device=self._device, dtype=torch.float).unsqueeze(-1) * per_frame_dist.unsqueeze(0)
            self.traj_position[env_ids[i], real_traj_index[env_ids[i]] + real_traj[env_ids[i]].shape[0]:, 0:2] += offset
            ####### compute the orientation  change ##########

        self.traj_position_flat = self.traj_position.reshape(-1, 3)
        return
    


class RealTrajGenerator():
    def __init__(self, num_envs, trajectory, origin_fps, device):
        self._device = device
        self._dt = 1.0 / origin_fps
        self._verts_flat = torch.from_numpy(trajectory).to(self._device).float().reshape(-1, 3)
        self._verts = self._verts_flat.view((num_envs, -1, 3))
    
    def reset(self, env_ids, init_pos):
        pass
        

    def get_num_verts(self):
        return self._verts.shape[1]

    def get_num_segs(self):
        return self.get_num_verts() - 1

    def get_num_envs(self):
        return self._verts.shape[0]

    def get_traj_duration(self):
        num_verts = self.get_num_verts()
        dur = num_verts * self._dt
        return  dur

    def get_traj_verts(self, traj_id):
        return self._verts[traj_id]

    def calc_pos(self, traj_ids, times):
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self._verts_flat[traj_ids * num_verts + seg_id0]
        pos1 = self._verts_flat[traj_ids * num_verts + seg_id1]

        lerp = lerp.unsqueeze(-1)
        pos = (1.0 - lerp) * pos0 + lerp * pos1
        return pos
    
    def get_velocity(self, traj_ids, times):
        pos_total = self.calc_pos(traj_ids.flatten(), times.flatten())
        pos_total = pos_total.reshape(traj_ids.shape[0], times.shape[1], pos_total.shape[1])
        pos = pos_total[:, 0]
        pos_next = pos_total[:, -1]
        vel = (pos_next - pos) / (times[:, -1] - times[:, 0]).unsqueeze(-1)
        return vel
    
    def get_init_pos(self, traj_ids):
        return self.calc_pos(traj_ids, torch.zeros_like(traj_ids, device=self._device))