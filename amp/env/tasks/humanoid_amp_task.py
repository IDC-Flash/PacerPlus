# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

import env.tasks.humanoid_amp as humanoid_amp

class HumanoidAMPTask(humanoid_amp.HumanoidAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._enable_task_obs = cfg["env"]["enableTaskObs"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.has_task = True
        return


    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_task_obs):
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size

    def get_task_obs_size(self):
        return 0

    def get_task_obs_size_detail(self):
        return NotImplemented

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_task()
        return

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return

    def _update_task(self):
        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        if len(env_ids) > 0:
            self._state_reset_happened = True
            self.obs_hist_buf[env_ids] *= 0
            self.temporal_obs_buf[env_ids] *= 0
            self.action_hist_buf[env_ids] *= 0
            self.reward_hist_buf[env_ids] *= 0
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._reset_task(env_ids)
            self._compute_observations(env_ids)
        self._init_amp_obs(env_ids)
        return

    def _reset_task(self, env_ids):
        return

    def _compute_observations(self, env_ids=None):
        # env_ids is used for resetting
        humanoid_obs = self._compute_humanoid_obs(env_ids)

        if (self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([humanoid_obs, task_obs], dim=-1)
        else:
            obs = humanoid_obs
        if self.has_flip_observation:
            flip_obs = self._compute_flip_humanoid_obs(env_ids)
            
            if (self._enable_task_obs):
                flip_task_obs = self._compute_flip_task_obs(task_obs, env_ids) 
                flip_obs = torch.cat([flip_obs, flip_task_obs], dim=-1)
                
            if (env_ids is None):
                self._flip_obs_buf[:] = flip_obs
            else:
                self._flip_obs_buf[env_ids] = flip_obs

        #### add by jingbo, resister obs histrory for transformer
        self.register_obs_hist(env_ids, obs) ### for all observation
        self.register_obs_buf(env_ids, humanoid_obs) ### for humanoid observation

        if (env_ids is None):
            if self._temporal_output:
                obs = torch.cat([self.obs_hist_buf, self.action_hist_buf], dim=1).reshape(self.num_envs, -1)
            if self.use_temporal_buf:
                obs = torch.cat([obs, self.temporal_obs_buf.reshape(self.num_envs, -1)], dim=-1)
            self.obs_buf[:] = obs
        else:
            if self._temporal_output:
                obs = torch.cat([self.obs_hist_buf[env_ids], self.action_hist_buf[env_ids]], dim=1).reshape(env_ids.shape[0], -1)
            if self.use_temporal_buf:
                obs = torch.cat([obs, self.temporal_obs_buf[env_ids].reshape(env_ids.shape[0], -1)], dim=-1)
            self.obs_buf[env_ids] = obs
        return

    def _compute_task_obs(self, env_ids=None):
        return NotImplemented

    def _compute_reward(self, actions):
        return NotImplemented

    def _draw_task(self):
        return
    
    