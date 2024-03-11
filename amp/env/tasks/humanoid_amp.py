# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from ast import Try
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
from enum import Enum
from matplotlib.pyplot import flag
import numpy as np
import torch
from torch import Tensor
from typing import Dict, Optional

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.humanoid import Humanoid, dof_to_obs, remove_base_rot, dof_to_obs_smpl
from env.util import gym_util
from amp.utils.motion_lib import MotionLib
from amp.utils.motion_lib_smpl import MotionLib as MotionLibSMPL
from poselib.poselib.skeleton.skeleton3d import SkeletonTree

from isaacgym.torch_utils import *
from utils import torch_utils

from uhc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
import gc
from amp.utils.flags import flags

HACK_MOTION_SYNC = False
# HACK_MOTION_SYNC = True
HACK_CONSISTENCY_TEST = False
HACK_OUTPUT_MOTION = False
HACK_OUTPUT_MOTION_ALL = False
SMPL_TO_H1 = [1, 2, 3, 5, 6, 7, 9, 15, 16, 20, 21]



class HumanoidAMP(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id,
                 headless):
        # jp hack
        if (HACK_MOTION_SYNC or HACK_CONSISTENCY_TEST):
            control_freq_inv = cfg["env"]["controlFrequencyInv"]
            self._motion_sync_dt = control_freq_inv * sim_params.dt
            cfg["env"]["controlFrequencyInv"] = 1
            cfg["env"]["pdControl"] = False

        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidAMP.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        ########### added by jingbo
        self._temporal_hist_length = cfg['env'].get("temporalHistLength", 1)
        self._temporal_output = cfg['env'].get("temporalOutput", False)
        self._temporal_buf_length = cfg['env'].get("temporalBufLength", 1)
        self.use_temporal_buf = self._temporal_buf_length > 1


        assert (self._num_amp_obs_steps >= 2)

        if ("enableHistObs" in cfg["env"]):
            self._enable_hist_obs = cfg["env"]["enableHistObs"]
        else:
            self._enable_hist_obs = False

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._state_reset_happened = False

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._motion_start_times = torch.zeros(self.num_envs).to(self.device)
        self._sampled_motion_ids = torch.zeros(self.num_envs).long().to(self.device)
        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        self.obs_hist_buf = torch.zeros(self.num_envs,
                                        self.num_obs,
                                        self._temporal_hist_length).to(self.device)
        self.action_hist_buf = torch.zeros(self.num_envs, 
                                           self.num_actions,
                                           self._temporal_hist_length).to(self.device)
        self.reward_hist_buf = torch.zeros(self.num_envs,
                                            1,
                                            self._temporal_hist_length).to(self.device)
        if self._temporal_output:
            self.num_obs = (self.num_actions + self.num_obs) * self._temporal_hist_length
        
        if self.use_temporal_buf:
            self.num_obs += self._num_self_obs * self._temporal_buf_length
            
        self.temporal_obs_buf = torch.zeros(self.num_envs,
                                            self._num_self_obs,
                                            self._temporal_buf_length).to(self.device) ######### used for record the self obs in temporal buffer

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs).to(self.device)
        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self._num_amp_obs_steps,
             self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        self._amp_obs_demo_buf = None
        self.start = True # camera flag
        self.ref_motion_cache = {}
        return

    def resample_motions(self):
        print("Partial solution, only resample motions...")
        self._motion_lib.load_motions(skeleton_trees = self.skeleton_trees, limb_weights = self.humanoid_limb_and_weights.cpu(), gender_betas = self.humanoid_betas.cpu()) # For now, only need to sample motions since there are only 400 hmanoids


    def register_obs_hist(self, env_ids, obs):
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        obs = obs.unsqueeze(-1)
        obs_hist_buf = torch.cat([self.obs_hist_buf[env_ids], obs], dim=-1)
        self.obs_hist_buf[env_ids] = obs_hist_buf[:, :, 1:]

    def register_obs_buf(self, env_ids, obs):
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        obs = obs.unsqueeze(-1)
        obs_hist_buf = torch.cat([self.temporal_obs_buf[env_ids], obs], dim=-1)
        self.temporal_obs_buf[env_ids] = obs_hist_buf[:, :, 1:]

    def register_action_hist(self, env_ids, actions):
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        actions = actions.unsqueeze(-1)
        action_hist_buf = torch.cat([self.action_hist_buf[env_ids, :, :-1], actions], dim=-1)
        action_hist_buf = action_hist_buf[..., 1:]
        action_hist_buf_pad = torch.zeros_like(action_hist_buf[..., :1])
        action_hist_buf = torch.cat([action_hist_buf, action_hist_buf_pad], dim=-1)
        self.action_hist_buf[env_ids] = action_hist_buf
    

    def register_reward_hist(self, env_ids,  rewards):
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        rewards = rewards.unsqueeze(-1)
        reward_hist_buf = torch.cat([self.reward_hist_buf[env_ids, :, :-1], rewards], dim=-1)
        reward_hist_buf = reward_hist_buf[..., 1:]
        reward_hist_buf_pad = torch.zeros_like(reward_hist_buf[..., :1])
        reward_hist_buf = torch.cat([reward_hist_buf, reward_hist_buf_pad], dim=-1)
        self.reward_hist_buf[env_ids] = reward_hist_buf

    def pre_physics_step(self, actions):
        env_ids = torch.arange(self.num_envs, device=self.device).to(self.device)
        if self._temporal_output:
            self.register_action_hist(env_ids, actions)

        if (HACK_MOTION_SYNC or HACK_CONSISTENCY_TEST):
            actions *= 0

        if flags.debug:
            actions *= 0

        super().pre_physics_step(actions)
        return

    def post_physics_step(self):
        super().post_physics_step();

        # jp hack
        if (HACK_CONSISTENCY_TEST):
            self._hack_consistency_test()
        elif (HACK_MOTION_SYNC):
            self._hack_motion_sync()

        if (HACK_OUTPUT_MOTION):
            self._hack_output_motion()

        self._update_hist_amp_obs()  # One step for the amp obs

        self._compute_amp_observations()


        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat  ## ZL: hooks for adding amp_obs for trianing

        # root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(
        #     torch.tensor([0]).to(self.device),
        #     torch.tensor([1.5]).to(self.device))
        # print(np.abs(dof_vel.cpu().numpy()).mean())
        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):
        # Creates the reference motion amp obs. For discrinminiator

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert (self._amp_obs_demo_buf.shape[0] == num_samples)

        motion_ids = self._motion_lib.sample_motions(num_samples)
        motion_times0 = self._sample_time(motion_ids)
        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(
            self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(
            -1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0):
        # Compute observation for the motion starting point
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1),[1, self._num_amp_obs_steps])

        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(
            0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        motion_times = torch.clamp(motion_times, 0)

        motion_res = self._get_smpl_state_from_motionlib_cache(
                motion_ids, motion_times)

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        amp_obs_demo = build_robot_amp_observation(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros(
            (num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float32)
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        self._num_amp_obs_per_step = 29

        if (self._enable_hist_obs):
            self._num_self_obs += self._num_amp_obs_steps * self._num_amp_obs_per_step

        return

    def _load_motion(self, motion_file):
        self._motion_lib = MotionLibSMPL(
                motion_file=motion_file,
                key_body_ids=self._contact_body_ids.cpu().numpy(),
                device=self.device)
        # load
        smpl_asset_path = os.getcwd()
        smpl_asset_file = 'amp/data/assets/mjcf/smpl_humanoid.xml'
        sk_tree = SkeletonTree.from_mjcf(os.path.join(smpl_asset_path, smpl_asset_file))
        skeleton_trees = [sk_tree] * self.num_envs
        self._motion_lib.load_motions(skeleton_trees = skeleton_trees, gender_betas = torch.zeros([self.num_envs, 11]).float(),
                                       limb_weights = torch.zeros([self.num_envs, 10]).float(), random_sample=not HACK_MOTION_SYNC)


        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        if len(env_ids) > 0:
            self._state_reset_happened = True
            self.obs_hist_buf[env_ids] *= 0
            self.action_hist_buf[env_ids] *= 0
            self.reward_hist_buf[env_ids] *= 0

        super()._reset_envs(env_ids)
        self._init_amp_obs(env_ids)

        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidAMP.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Start
              or self._state_init == HumanoidAMP.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert (
                False
            ), "Unsupported state initialization strategy: {:s}".format(
                str(self._state_init))
        return

    def _reset_default(self, env_ids):
        self._humanoid_root_states[
            env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _sample_time(self, motion_ids):
        return self._motion_lib.sample_time(motion_ids)

    def _get_smpl_state_from_motionlib_cache(self, motion_ids, motion_times):
        motion_times = torch.clamp(motion_times, 0)
        motion_res = self._motion_lib.get_motion_state_smpl(motion_ids, motion_times)

        ######## change smpl dof to h1
        dof_pos = motion_res["dof_pos"]
        dof_vel = motion_res["dof_vel"]

        dof_pos = dof_pos.reshape(-1, 23, 3)[:, SMPL_TO_H1, :]
        dof_vel = dof_vel.reshape(-1, 23, 3)[:, SMPL_TO_H1, :]
            
        dof_pos = torch_utils.exp_map_to_quat(dof_pos)
        dof_vel = torch_utils.exp_map_to_quat(dof_vel)

        B, N = dof_pos.shape[:2]
        dof_pos = dof_pos.reshape(B*N, 4)
        dof_pos = get_euler_xyz(dof_pos)
        dof_pos = (dof_pos.reshape(B, N, 3)) / (np.pi / 2)

        dof_pos = torch.cat(( dof_pos[:, 0, [2, 0, 1]], dof_pos[:, 1, 1:2], dof_pos[:, 2, 1:2],
                              dof_pos[:, 3, [2, 0, 1]], dof_pos[:, 4, 1:2], dof_pos[:, 5, 1:2], 
                              dof_pos[:, 6, 1:2],
                              dof_pos[:, 7, [1, 0, 2]], dof_pos[:, 8, 1:2],
                              dof_pos[:, 9,[1, 0, 2]], dof_pos[:, 10, 1:2],
                              ), dim=-1)
        
        dof_pos += self.default_dof_pos
        dof_pos[:, 12] += 90 / 360
        dof_pos[:, 14] += 90 / 360 
        dof_pos[:, 16] -= 90 / 360
        dof_pos[:, 18] -= 90 / 360

        dof_vel = dof_vel.reshape(B*N, 4)
        dof_vel = get_euler_xyz(dof_vel)
        dof_vel = dof_vel.reshape(B, N, 3) / (np.pi / 2)
        dof_vel = torch.cat(( dof_vel[:, 0, [2, 0, 1]], dof_vel[:, 1, 1:2], dof_vel[:, 2, 1:2],
                              dof_vel[:, 3, [2, 0, 1]], dof_vel[:, 4, 1:2], dof_vel[:, 5, 1:2], 
                              dof_vel[:, 6, 1:2],
                              dof_vel[:, 7, [1, 0, 2]], dof_vel[:, 8, 1:2],
                              dof_vel[:, 9,[1, 0, 2]], dof_vel[:, 10, 1:2],
                              ), dim=-1)
        #dof_pos[:, :10] += self.default_do    
        motion_res["dof_pos"] = dof_pos
        motion_res["dof_vel"] = dof_vel
        
        return motion_res

    def _sample_ref_state(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)

        if (self._state_init == HumanoidAMP.StateInit.Random
                or self._state_init == HumanoidAMP.StateInit.Hybrid):
            motion_times = self._sample_time(motion_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert (
                False
            ), "Unsupported state initialization strategy: {:s}".format(
                str(self._state_init))

        motion_res = self._get_smpl_state_from_motionlib_cache(motion_ids, motion_times)

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot = self._sample_ref_state(env_ids)

        if flags.debug:
            root_pos[..., 2] += 0.5

        if flags.fixed:
            x_grid, y_grid = torch.meshgrid(torch.arange(64), torch.arange(64))
            root_pos[:, 0], root_pos[:, 1] = x_grid.flatten()[env_ids] * 2, y_grid.flatten()[env_ids] * 2

        self._set_env_state(env_ids=env_ids,
                            root_pos=root_pos,
                            root_rot=root_rot,
                            dof_pos=dof_pos,
                            root_vel=root_vel,
                            root_ang_vel=root_ang_vel,
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        self._motion_start_times[env_ids] = motion_times
        self._sampled_motion_ids[env_ids] = motion_ids
        if flags.follow:
            self.start = True  ## Updating camera when reset
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs),
                             device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]

        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _compute_humanoid_obs(self, env_ids=None):
        obs = super()._compute_humanoid_obs(env_ids)

        if (self._enable_hist_obs):
            if (env_ids is None):
                hist_obs = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
            else:
                hist_obs = self._amp_obs_buf[env_ids].view(
                    -1, self.get_num_amp_obs())

            obs = torch.cat([obs, hist_obs], dim=-1)

        return obs

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids,
                                   self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)

        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1),[1, self._num_amp_obs_steps-1])
        motion_times = motion_times.unsqueeze(-1)

        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        motion_res = self._get_smpl_state_from_motionlib_cache(motion_ids, motion_times)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        amp_obs_demo = build_robot_amp_observation(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel)


        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return

    def _set_env_state(self,
                       env_ids,
                       root_pos,
                       root_rot,
                       dof_pos,
                       root_vel,
                       root_ang_vel,
                       dof_vel,
                       rigid_body_pos=None,
                       rigid_body_rot=None):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        if (not rigid_body_pos is None) and (not rigid_body_rot is None):
            self._rigid_body_pos[env_ids] = rigid_body_pos
            self._rigid_body_rot[env_ids] = rigid_body_rot

            self._rigid_body_vel[env_ids] = 0
            self._rigid_body_ang_vel[env_ids] = 0
            self._reset_rb_pos = self._rigid_body_pos[env_ids].clone()
            self._reset_rb_rot = self._rigid_body_rot[env_ids].clone()

        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            try:
                self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0:(self._num_amp_obs_steps - 1)]
            except:
                self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0:(self._num_amp_obs_steps - 1)].clone()
        else:
            self._hist_amp_obs_buf[env_ids] = self._amp_obs_buf[env_ids, 0:(
                self._num_amp_obs_steps - 1)]
        return

    def _compute_amp_observations(self, env_ids=None):

        if (env_ids is None):
            root_pos = self._base_pos.clone()
            root_rot = self._base_quat.clone()
            root_vel = self._humanoid_root_states[:, 7:10].clone()
            root_ang_vel = self._humanoid_root_states[:, 10:13].clone()
            dof_pos = self._dof_pos.clone()
            dof_vel = self._dof_vel.clone()
            self._curr_amp_obs_buf[:] = build_robot_amp_observation(
                    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel)
        elif len(env_ids) == 0:
            return
        else:
            root_pos = self._base_pos[env_ids].clone()
            root_rot = self._base_quat[env_ids].clone()
            dof_pos = self._dof_pos[env_ids].clone()
            dof_vel = self._dof_vel[env_ids].clone()
            root_vel = self._humanoid_root_states[env_ids, 7:10].clone()
            root_ang_vel = self._humanoid_root_states[env_ids, 10:13].clone()

            self._curr_amp_obs_buf[env_ids] = build_robot_amp_observation(
                    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel)

        return



#####################################################################
###=========================jit functions=========================###
#####################################################################

#@torch.jit.script
def build_robot_amp_observation(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_h = root_pos[:, 2:3]
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    local_root_vel = torch_utils.my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = torch_utils.my_quat_rotate(heading_rot, root_ang_vel)
    remove_id = [4, 9]
    selected_id = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    obs = torch.cat((root_rot_obs, local_root_vel, local_root_ang_vel, dof_pos[..., selected_id]), dim=-1)
                    
    return obs

#@torch.jit.script
def get_euler_xyz(q):
    # type: (Tensor)  -> Tensor
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)