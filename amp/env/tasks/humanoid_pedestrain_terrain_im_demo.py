# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from shutil import ExecError
import torch
import glob
import os
import pickle
import numpy as np
import env.util.traj_generator as traj_generator
import joblib
import amp.env.tasks.humanoid_pedestrain_terrain_im as humanoid_pedestrain_terrain_im
from isaacgym import gymapi
from isaacgym.torch_utils import *
from env.tasks.humanoid import dof_to_obs
from env.tasks.humanoid_amp import HumanoidAMP, remove_base_rot
from amp.utils.flags import flags
from utils import torch_utils
from utils import konia_transform
from isaacgym import gymtorch
from poselib.poselib.core.rotation3d import quat_inverse, quat_mul
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot
import matplotlib.pyplot as plt
from amp.utils.draw_utils import agt_color
from amp.utils.motion_lib_smpl import MotionLib as MotionLibSMPL
from amp.env.tasks.humanoid_traj import compute_location_reward


HACK_MOTION_SYNC = False

class HumanoidPedestrianTerrainImDemo(humanoid_pedestrain_terrain_im.HumanoidPedestrianTerrainIm):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.real_trajectory = cfg['args'].real_trajectory
        self.full_body_motion = cfg['env'].get('full_body_motion', False)
        self.use_different_motion_file = cfg["env"].get("use_different_motion_file", True)
        self.reset_buffer = cfg["env"].get("reset_buffer", 0)
        print('use_different_motion_file', self.use_different_motion_file)
        #### input tracking mask into the observation
        self.has_tracking_mask = cfg["env"].get("has_tracking_mask", False)
        self.has_tracking_mask_obs = cfg["env"].get("has_tracking_mask_obs", False) and self.has_tracking_mask

        self.only_track_upper_body = cfg["env"].get("only_tracking_upper_body", False)

        self.use_lower_body_ref_dof = cfg["env"].get("use_lower_body_ref_dof", False)
        self.resample_motion_content = False
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self._track_bodies = cfg["env"].get("trackBodies", self._full_track_bodies)
        self._track_bodies_id = self._build_key_body_ids_tensor(self._track_bodies)
        self._full_track_bodies_id = self._build_key_body_ids_tensor(self._full_track_bodies)
        self._reset_bodies = cfg["env"].get("resetBodies", self._full_track_bodies.copy())
        if self.remove_foot_reset_im:
            self._reset_bodies.remove('L_Ankle');self._reset_bodies.remove('R_Ankle');
            if not self.remove_toe_im:
                self._reset_bodies.remove('L_Toe');self._reset_bodies.remove('R_Toe');
        self._reset_body_ids = self._build_key_body_ids_tensor(self._reset_bodies)
        self._pd_use_ref_dof = cfg["env"].get("pd_use_ref_dof", False)
        self.reward_raw = torch.zeros((self.num_envs, 3)).to(self.device)
        self.terminate_dist = cfg['env'].get('terminate_dist', 0.4), 
        self.use_imitation_reset = cfg['env'].get('use_imitation_reset', False)
        self.use_imitation_reset = False
        
        mld_motion_file = self.cfg['env']['demo_motion_file']
        self._load_demo_motion(mld_motion_file)
        self.previous_heading = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.imitation_ref_motion_cache = {}
        self.d3_visible = torch.zeros((self.num_envs), dtype=torch.int, device=self.device)
        self.mld_motion_cache = {}
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 2 * self._num_traj_samples
            if self.terrain_obs:
                if self.velocity_map:
                    obs_size += self.num_height_points * 3
                else:
                    obs_size += self.num_height_points
            if self._divide_group and self._group_obs:
                obs_size += 5 * 11 * 3
            obs_size +=  249  + 1 + 63
            if self.has_tracking_mask_obs:
                obs_size += 24
        return obs_size
    
    def get_task_obs_size_detail(self):
        task_obs_detail = []
        if (self._enable_task_obs):
            task_obs_detail.append(["traj", 2 * self._num_traj_samples])

        if self.terrain_obs:
            if self.velocity_map:
                task_obs_detail.append(["heightmap_velocity", self.num_height_points * 3])
            else:
                task_obs_detail.append(["heightmap", self.num_height_points])

        if self._divide_group and self._group_obs:
            task_obs_detail.append(["people", 5 * 11 * 3])
        task_obs_detail.append(["imitation_target", 249 + 63])
        task_obs_detail.append(["imitation_target_visible", 1])
        task_obs_detail.append(["tracking_mask", 24])

        return task_obs_detail

        
    def build_body_tracking_mask(self, env_ids):
        ### build tracking_mask
        if self.has_tracking_mask:
            if not self.only_track_upper_body:
                tracking_mask = torch.zeros((env_ids.shape[0], self.num_bodies), dtype=torch.int, device=self.device)
                selecte_num = torch.randint(self.num_bodies//2, self.num_bodies, (1,)).item()
                selected_idx = torch.randperm(self.num_bodies)[:selecte_num].to(self.device)
                tracking_mask[:, selected_idx] = 1
                tracking_mask[:, 0] = 1
                self.tracking_mask = tracking_mask.unsqueeze(-1)
            else:
                # tracking_mask = torch.zeros((env_ids.shape[0], self.num_bodies), dtype=torch.int, device=self.device)
                # selecte_num = torch.randint(self.num_bodies//2, self.num_bodies, (1,)).item()
                # selected_idx = torch.randperm(self.num_bodies)[:selecte_num].to(self.device)
                # tracking_mask[:, selected_idx] = 1
                # tracking_mask[:, 0] = 1
                # self.tracking_mask = tracking_mask.unsqueeze(-1)
                self._upper_track_bodies_id = self._build_key_body_ids_tensor(self._upper_track_bodies)
                self.tracking_mask = torch.zeros((env_ids.shape[0], self.num_bodies, 1), device=self.device, dtype=torch.int)
                self.tracking_mask[:, self._upper_track_bodies_id] = 1
                self.tracking_mask[:, 0] = 1
        else:
            self.tracking_mask = torch.ones((env_ids.shape[0], self.num_bodies, 1), dtype=torch.int, device=self.device)

        if flags.test:
            self._upper_track_bodies_id = self._build_key_body_ids_tensor(self._upper_track_bodies)
            self.tracking_mask = torch.zeros((env_ids.shape[0], self.num_bodies, 1), device=self.device, dtype=torch.int)
            self.tracking_mask[:, self._upper_track_bodies_id] = 1
            self.tracking_mask[:, 0] = 1
            if self.full_body_motion:
                self.tracking_mask[:, :8] = 1

    def _load_demo_motion(self, mld_motion_file):
        assert (self._dof_offsets[-1] == self.num_dof)

        self._demo_motion_lib = MotionLibSMPL(
            motion_file=mld_motion_file,
            key_body_ids=self._key_body_ids.cpu().numpy(),
            device=self.device, masterfoot_conifg=self._masterfoot_config,
            min_length=self._min_motion_len)
        self._demo_motion_lib.load_motions(skeleton_trees = self.skeleton_trees, gender_betas = self.humanoid_betas.cpu(),
            limb_weights = self.humanoid_limb_and_weights.cpu(), random_sample=not flags.test)

        self.reference_start_index = torch.zeros((self.num_envs), dtype=torch.int, device=self.device)
        self.reference_end_index = torch.zeros((self.num_envs), dtype=torch.int, device=self.device)
        self.reference_length = torch.zeros((self.num_envs), dtype=torch.int, device=self.device)


        self.build_body_tracking_mask(torch.arange(self.num_envs, device=self.device, dtype=torch.int))
        return

    def _build_traj_generator(self):
        if not self.real_trajectory:
            num_envs = self.num_envs
            episode_dur = self.max_episode_length * self.dt
            num_verts = 101
            dtheta_max = 2.0
            self._traj_gen = traj_generator.TrajGenerator(num_envs, episode_dur, num_verts,
                                                        self.device, dtheta_max,
                                                        self._speed_min, self._speed_max,
                                                        self._accel_max, self._sharp_turn_prob)
        else:
            pass
    
    
    
    def _reset_task(self, env_ids):
        if not flags.server_mode:
            root_pos = self._humanoid_root_states[env_ids, 0:3]
            if not self.full_body_motion:
                self._traj_gen.reset(env_ids, root_pos)
        
        #### obtain begin and end frame index
        reference_frames = self._demo_motion_lib.get_motion_num_frames(env_ids)
        traj_length = self.max_episode_length
        for i in range(env_ids.shape[0]):
            if not hasattr(self, 'time_info'):
                self.reference_start_index[i] = torch.randint(30, (traj_length - reference_frames[i]) //4, (1,), device=self.device)
                self.reference_end_index[i] = self.reference_start_index[i] + reference_frames[i]
                self.reference_length[i] = reference_frames[i]
            else:
                begin_time = self.time_info['begin_time']
                end_time = self.time_info['end_time']
                if begin_time is not None:
                    begin_time = int(begin_time)
                    begin_time = np.clip(0, self.max_episode_length, begin_time)
                    self.reference_start_index[i] = begin_time
                if end_time is not None:
                    end_time = int(end_time)
                    end_time = np.clip(0, self.max_episode_length, end_time)
                    self.reference_end_index[i] = self.reference_start_index[i] + reference_frames[i]
                    self.reference_length[i] = reference_frames[i] 

        if self.full_body_motion:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            root_pos = self._humanoid_root_states[:, 0:3]
            reference_motion_traj = self._demo_motion_lib.gts[:, 0].clone()
            self.motion_begin = torch.tensor(self.reference_start_index).to(self.device).int() + self._demo_motion_lib.length_starts
            self.motion_end = self.motion_begin + torch.tensor(self.reference_length).to(self.device).int()
            self.reference_motion_traj = [reference_motion_traj[self.motion_begin[i]:self.motion_end[i]] for i in range(self.num_envs)]

            if not flags.server_mode:
                self._traj_gen.reset(env_ids, root_pos, self.reference_motion_traj, self.reference_start_index)
            else:
                self._traj_gen.insert_real_trajectory(env_ids, self.reference_motion_traj, self.reference_start_index)

        return

    def resample_motions(self):
        print("Partial solution, only resample motions...")
        self._demo_motion_lib.load_motions(skeleton_trees = self.skeleton_trees, limb_weights = self.humanoid_limb_and_weights.cpu(), gender_betas = self.humanoid_betas.cpu()) # For now, only need to sample motions since there are only 400 hmanoids
        self.reset()

    def resample_motions_for_on_demand_control(self):
        print("On Demand Control !!!")
        motion_content_dict = {
            'A man is walking, calling a phone and waving his right hand.': [0, 10],
            'A man is walking, filling exciting and putting all his hands up his head.':[11, 20],
            'A man is walking, calling a phone and waving his left hand.': [21, 30],
            'Create a pedestrian motion of a person walking briskly along the sidewalk.':[31, 35],
            'Create a pedestrian motion of a person walking while engaged in conversation with another pedestrian.': [36, 40],
            'Generate a pedestrian motion of someone reacting to sudden changes in traffic or road conditions.': [41, 45],
            'Create a pedestrian motion of a person interacting with a street performer or vendor.':[45, 50]
        }
        if hasattr(self, 'motion_info') and self.resample_motion_content:
            begin_idx, end_idx = motion_content_dict[self.motion_info['change_motion']]
            idx = np.random.randint(begin_idx, end_idx)
            self._demo_motion_lib.load_motions(
                skeleton_trees = self.skeleton_trees,
                limb_weights = self.humanoid_limb_and_weights.cpu(),
                gender_betas = self.humanoid_betas.cpu(),
                random_sample=True,
                start_idx=idx
            )
            self.resample_motion_content = False
        if hasattr(self, 'body_info') and self.change_body_content:
            body_part = self.body_info['selected_body']
            if body_part == 'Whole Body':
                #self._upper_track_bodies_id = self._build_key_body_ids_tensor(self._upper_track_bodies)
                self.tracking_mask = torch.ones((self.num_envs, self.num_bodies, 1), device=self.device, dtype=torch.int)
            else:
                self._upper_track_bodies_id = self._build_key_body_ids_tensor(self._upper_track_bodies)
                self.tracking_mask = torch.zeros((self.num_envs, self.num_bodies, 1), device=self.device, dtype=torch.int)
                self.tracking_mask[:, self._upper_track_bodies_id] = 1
                self.tracking_mask[:, 0] = 1
                if body_part == 'Left Arm':
                    self.tracking_mask[:, 19:24] = 0
                elif body_part == 'Right Arm':  
                    self.tracking_mask[:, 14:20] = 0
            self.change_body_content = False
        self.reset()
        

    def _get_mld_state_from_motionlib(self, motion_ids, motion_times):
        if self.resample_motion_content:
            return self.mld_motion_cache
        motion_res = self._demo_motion_lib.get_motion_state_smpl(motion_ids, motion_times)
        self.mld_motion_cache.update(motion_res)
        return self.mld_motion_cache



    def use_buffer_state(self, body_pos, body_rot, dof_pos, ref_body_pos, ref_body_rot, ref_dof_pos, exp):
        body_pos = body_pos.clone() - body_pos[:, :1, :]
        ref_body_pos = ref_body_pos.clone() - ref_body_pos[:, :1, :]
        blend_ref_body_pos = body_pos * exp + ref_body_pos * (1-exp)

        blend_ref_rot = torch_utils.slerp(body_rot, ref_body_rot, (1-exp))


        dof_pos = dof_pos.reshape(dof_pos.shape[0], -1, 3)
        ref_dof_pos = ref_dof_pos.reshape(ref_dof_pos.shape[0], -1, 3)
        dof_pos_quat = torch_utils.exp_map_to_quat(dof_pos)
        ref_dof_pos_quat = torch_utils.exp_map_to_quat(ref_dof_pos)
        blend_ref_dof_pos_quat = torch_utils.slerp(dof_pos_quat, ref_dof_pos_quat, (1-exp))
        blend_ref_dof_pos = torch_utils.quat_to_exp_map(blend_ref_dof_pos_quat)
        blend_ref_dof_pos = blend_ref_dof_pos.reshape(blend_ref_dof_pos.shape[0], -1)
        return blend_ref_body_pos, blend_ref_rot, blend_ref_dof_pos

        
    def _compute_task_obs(self, env_ids=None):
        # Compute task observations (terrain, trajectory, self state)
        basic_obs = super(humanoid_pedestrain_terrain_im.HumanoidPedestrianTerrainIm, self)._compute_task_obs(env_ids)
        # Compute IM observations
        if (env_ids is None):
            body_pos = self._rigid_body_pos.clone()
            body_rot = self._rigid_body_rot.clone()
            body_vel = self._rigid_body_vel.clone()
            body_ang_vel = self._rigid_body_ang_vel.clone()
            body_dof = self._dof_pos.clone()
            env_ids = torch.arange(self.num_envs,
                               dtype=torch.long,
                               device=self.device)
        else:
            body_pos = self._rigid_body_pos[env_ids].clone()
            body_rot = self._rigid_body_rot[env_ids].clone()
            body_vel = self._rigid_body_vel[env_ids].clone()
            body_dof = self._dof_pos[env_ids].clone()
            body_ang_vel = self._rigid_body_ang_vel[env_ids].clone()

        
        ######### we need a flag for this observation
        ######### if flag == 1, we need to compute the observation
        ######### if flag == 0, we need do not need the imitation target in the observation

        # ref_start = torch.tensor(self.reference_start_index).to(self.device)
        # ref_length = torch.tensor(self.reference_length).to(self.device)
        motion_times = (self.progress_buf[env_ids] + 1) * self.dt
        reference_time = (self.progress_buf[env_ids] + 1 - self.reference_start_index[env_ids]) * self.dt
        reference_time = torch.clamp(reference_time, min=0, max=self.max_episode_length * self.dt)

        # if env_ids is not None and env_ids.shape[0] != self.num_envs:
        #     print(env_ids)
        motion_res = self._get_mld_state_from_motionlib(env_ids, reference_time)
        time_steps = 1
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"]
        
        buffer = torch.min(torch.ones_like(self.reference_start_index[env_ids]) * 0, self.reference_start_index[env_ids])
        d3_visible = torch.zeros((env_ids.shape[0]), dtype=torch.int, device=self.device)
        mask = ((self.progress_buf[env_ids] + 1) >= self.reference_start_index[env_ids] - buffer) & ((self.progress_buf[env_ids] + 1) < self.reference_end_index[env_ids])
        d3_visible[mask] = 1


        buffer_mask = ((self.progress_buf[env_ids] + 1) >= self.reference_start_index[env_ids] - buffer) & ((self.progress_buf[env_ids] + 1) < self.reference_start_index[env_ids])
        if buffer_mask.sum() > 0:
            ref_rb_pos_blend, ref_rb_rot_blend, dof_pos_blend = self.use_buffer_state(body_pos[buffer_mask], body_rot[buffer_mask], body_dof[buffer_mask], 
                                                                                      ref_rb_pos[buffer_mask], ref_rb_rot[buffer_mask], dof_pos[buffer_mask], 
                                                                                      (self.reference_start_index[env_ids] - self.progress_buf[env_ids] - 1) / buffer)
            ref_rb_pos[buffer_mask] = ref_rb_pos_blend
            ref_rb_rot[buffer_mask] = ref_rb_rot_blend
            dof_pos[buffer_mask] = dof_pos_blend


        ## clone data from motion library
        ref_rb_pos = ref_rb_pos.clone()
        ref_rb_rot = ref_rb_rot.clone()
        ref_body_vel = ref_body_vel.clone()
        ref_traj_pos = self._traj_gen.calc_pos(env_ids,motion_times)
        ref_rb_pos[:, :, :2] -= ref_rb_pos[:, 0, :2].clone().unsqueeze(1)
        ref_rb_pos[:, :, 0:2] += ref_traj_pos[:, :2].unsqueeze(1)

        #if not self.full_body_motion:
        ## change heading via trajectory
        ## step-1 obatin trajectory
        ## step-2 compute relative heading
        ## step-3 change reference pos and rot
        trajectory = self.traj_samples.clone()
        traj_begin, traj_end = trajectory[:, 0], trajectory[:, 2]
        traj_heading = compute_traj_heading(traj_begin, traj_end)
        motion_heading = compute_motion_heading(ref_rb_rot[:, 0])
        ref_rb_pos, ref_rb_rot = process_imitation_target_with_heading(traj_heading, motion_heading, ref_rb_pos, ref_rb_rot)

        ref_root_rot = ref_rb_rot[:, 0, :].clone()
        ref_root_pos = ref_rb_pos[:, 0, :].clone()
        ## changed height
        root_state = torch.cat([ref_root_pos, ref_root_rot], dim=-1)
        center_heights = self.get_center_heights(root_states=root_state, env_ids=env_ids)
        center_heights = center_heights.mean(dim=-1, keepdim=True)
        ref_rb_pos[..., 2] += center_heights

        ########### update for visualization
        if flags.test:  
            self.im_ref_rb_target_pos = ref_rb_pos.clone()[:, ]

        
        tracking_mask = self.tracking_mask[env_ids].clone()
        dof_tracking_mask = tracking_mask[:, self._track_bodies_id, :].clone()
        dof_tracking_mask = dof_tracking_mask[:, 1:]

        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]
        body_pos_subset = body_pos[..., self._track_bodies_id, :]  * tracking_mask[:, self._track_bodies_id, :]
        body_rot_subset = body_rot[..., self._track_bodies_id, :] * tracking_mask[:, self._track_bodies_id, :]
        body_dof = body_dof.reshape(body_dof.shape[0], -1, 3)
        body_dof_subset = body_dof[..., [i-1 for i in self._track_bodies_id[1:]], :] * dof_tracking_mask ### dof does not have root
        body_dof_subset = body_dof_subset.reshape(body_dof_subset.shape[0], -1)


        ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :] * tracking_mask[:, self._track_bodies_id, :]
        ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :] * tracking_mask[:, self._track_bodies_id, :]
        dof_pos = dof_pos.reshape(dof_pos.shape[0], -1, 3)
        dof_pos_subset = dof_pos[..., [i-1 for i in self._track_bodies_id[1:]], :] * dof_tracking_mask ### dof does not have root
        dof_pos_subset = dof_pos_subset.reshape(dof_pos_subset.shape[0], -1)

        obs = compute_imitation_observations(root_pos, root_rot, body_pos_subset, body_rot_subset, body_dof_subset, ref_rb_pos_subset, ref_rb_rot_subset, 
                                             dof_pos_subset, d3_visible, tracking_mask[:, self._track_bodies_id, :], time_steps, self._has_upright_start)
        
        obs = torch.cat([basic_obs, obs, d3_visible.unsqueeze(-1)], dim=-1)
        if self.has_tracking_mask_obs:
            obs = torch.cat([obs, tracking_mask.squeeze(-1) * d3_visible[:, None]], dim=-1)


        if env_ids is None:
            use_env_ids = False
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            use_env_ids = True

        if not use_env_ids:
            self.d3_visible = d3_visible
        else:
            self.d3_visible[env_ids] = d3_visible

        return obs

    def _compute_flip_task_obs(self, normal_task_obs, env_ids):

        # location_obs  20
        # Terrain obs: self.num_terrain_obs
        # group obs
        basic_obs = super(humanoid_pedestrain_terrain_im.HumanoidPedestrianTerrainIm, self)._compute_flip_task_obs(normal_task_obs, env_ids)
        if (env_ids is None):
            body_pos = self._rigid_body_pos.clone()
            body_rot = self._rigid_body_rot.clone()
            body_vel = self._rigid_body_vel.clone()
            body_ang_vel = self._rigid_body_ang_vel.clone()
            body_dof = self._dof_pos.clone()
            env_ids = torch.arange(self.num_envs,
                               dtype=torch.long,
                               device=self.device)
        else:
            body_pos = self._rigid_body_pos[env_ids].clone()
            body_rot = self._rigid_body_rot[env_ids].clone()
            body_vel = self._rigid_body_vel[env_ids].clone()
            body_dof = self._dof_pos[env_ids].clone()
            body_ang_vel = self._rigid_body_ang_vel[env_ids]

        curr_gender_betas = self.humanoid_betas[env_ids]
        
        ######### we need a flag for this observation
        ######### if flag == 1, we need to compute the observation
        ######### if flag == 0, we need do not need the imitation target in the observation

        motion_times = (self.progress_buf[env_ids] + 1) * self.dt
        reference_time = (self.progress_buf[env_ids] + 1 - self.reference_start_index[env_ids]) * self.dt
        reference_time = torch.clamp(reference_time, min=0, max=self.max_episode_length * self.dt)
        motion_res = self._get_mld_state_from_motionlib(env_ids, reference_time)
        time_steps = 1
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"]

        d3_visible = torch.zeros((env_ids.shape[0]), dtype=torch.int, device=self.device)
        mask = ((self.progress_buf[env_ids] + 1) >= self.reference_start_index[env_ids]) & ((self.progress_buf[env_ids] + 1) < self.reference_end_index[env_ids])
        d3_visible[mask] = 1

        ################## prepare for imitation
        root_pos = root_pos.clone()
        ref_rb_pos = ref_rb_pos.clone()
        ref_rb_rot = ref_rb_rot.clone()
        ref_body_vel = ref_body_vel.clone()
        root_states = torch.cat([root_pos, root_rot], dim=-1).clone()
        center_heights = self.get_center_heights(root_states=root_states, env_ids=env_ids)
        center_heights = center_heights.mean(dim=-1, keepdim=True)
        ref_rb_pos[..., 2] += center_heights

        ################## Flip left to right
        body_pos[..., 1] *= -1 # position
        body_pos = body_pos[..., self.left_to_right_index, :]
        body_rot[..., 0] *= -1 # angular rotation, global
        body_rot[..., 2] *= -1
        body_rot = body_rot[..., self.left_to_right_index, :]

        body_dof = body_dof.reshape(body_dof.shape[0], -1, 3)
        body_dof[..., 0] *= -1 # dof rotation local
        body_dof[..., 2] *= -1
        body_dof = body_dof[..., self.left_to_right_index_action, :]

        #######
        ref_rb_pos[..., 1] *= -1 # position
        ref_rb_pos = ref_rb_pos[..., self.left_to_right_index, :]
        ref_rb_rot[..., 0] *= -1
        ref_rb_rot[..., 2] *= -1
        ref_rb_rot = ref_rb_rot[..., self.left_to_right_index, :]

        dof_pos = dof_pos.reshape(dof_pos.shape[0], -1, 3)
        dof_pos[..., 0] *= -1 # dof rotation local
        dof_pos[..., 2] *= -1
        dof_pos = dof_pos[..., self.left_to_right_index_action, :]

        tracking_mask_flip = self.tracking_mask[env_ids].clone()
        tracking_mask_flip = tracking_mask_flip[..., self.left_to_right_index, :]
        dof_tracking_mask_flip = tracking_mask_flip[:, self._track_bodies_id, :].clone()
        dof_tracking_mask_flip = dof_tracking_mask_flip[:, 1:]

        ################## Flip left to right
        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]
        body_pos_subset = body_pos[..., self._track_bodies_id, :]  * tracking_mask_flip[:, self._track_bodies_id, :]
        body_rot_subset = body_rot[..., self._track_bodies_id, :] * tracking_mask_flip[:, self._track_bodies_id, :]
        body_dof = body_dof.reshape(body_dof.shape[0], -1, 3)
        body_dof_subset = body_dof[..., [i-1 for i in self._track_bodies_id[1:]], :] * dof_tracking_mask_flip ### dof does not have root
        body_dof_subset = body_dof_subset.reshape(body_dof_subset.shape[0], -1)

        ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :] * tracking_mask_flip[:, self._track_bodies_id, :]
        ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :] *tracking_mask_flip[:, self._track_bodies_id, :]

        dof_pos_subset = dof_pos[..., [i-1 for i in self._track_bodies_id[1:]], :] * dof_tracking_mask_flip ### dof does not have root
        dof_pos_subset = dof_pos_subset.reshape(dof_pos_subset.shape[0], -1)

        obs = compute_imitation_observations(root_pos, root_rot, body_pos_subset, body_rot_subset, body_dof_subset, ref_rb_pos_subset, ref_rb_rot_subset,
                                              dof_pos_subset, d3_visible, tracking_mask_flip[:, self._track_bodies_id, :], time_steps, self._has_upright_start)

        obs = torch.cat([basic_obs, obs, d3_visible.unsqueeze(-1)], dim=-1)
        if self.has_tracking_mask_obs:
            obs = torch.cat([obs, tracking_mask_flip.squeeze(-1) * d3_visible[:, None]], dim=-1)
        return obs


    def _compute_reward(self, actions):
        w_location, w_imitation = self.task_reward_specs['w_location'], self.task_reward_specs['w_imitation']

        root_pos = self._humanoid_root_states[..., 0:3]

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self._traj_gen.calc_pos(env_ids, time)

        location_reward = compute_location_reward(root_pos, tar_pos)
        im_reward = self._compute_im_rewards(actions)
        self.rew_buf[:] = location_reward * w_location + im_reward * w_imitation

        power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim = -1)
        power_reward = -self.power_coefficient * power

        if self.power_reward:
            self.rew_buf[:] += power_reward
        self.reward_raw[:] = torch.cat([location_reward[:, None], im_reward[:, None], power_reward[:, None]], dim = -1)
    
        return


    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot = self._sample_ref_state(env_ids)
        ## Randomrized location setting
        if not self.real_trajectory:
            new_root_xy = self.terrain.sample_valid_locations(self.num_envs, env_ids)
        else:
            new_root_xy = self._traj_gen.calc_pos(env_ids, torch.zeros_like(env_ids))[:, :2]

        if flags.fixed:
            new_root_xy[:, 0], new_root_xy[:, 1] = 10 + env_ids * 3, 10


        if flags.server_mode:
            new_traj = self._traj_gen.input_new_trajs(env_ids)
            new_root_xy[:, 0], new_root_xy[:, 1] = new_traj[:, 0, 0], new_traj[:, 0,  1]


        diff_xy = new_root_xy - root_pos[:, 0:2]
        root_pos[:, 0:2] = new_root_xy
        root_states = torch.cat([root_pos, root_rot], dim=1)
        center_height = self.get_center_heights(root_states, env_ids=env_ids).mean(dim=-1)

        root_pos[:, 2] += center_height
        key_pos[..., 0:2] += diff_xy[:, None, :]
        key_pos[...,  2] += center_height[:, None]

        rb_pos[..., 0:2] += diff_xy[:, None, :]
        rb_pos[..., 2] += center_height[:, None]

        self._set_env_state(env_ids=env_ids,
                            root_pos=root_pos,
                            root_rot=root_rot,
                            dof_pos=dof_pos,
                            root_vel=root_vel,
                            root_ang_vel=root_ang_vel,
                            dof_vel=dof_vel,
                            rigid_body_pos=rb_pos,
                            rigid_body_rot=rb_rot)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        if flags.follow:
            self.start = True  ## Updating camera when reset

        return

    def _compute_im_rewards(self, actions):

        env_ids = torch.arange(self.num_envs,
                               dtype=torch.long,
                               device=self.device)
        body_pos = self._rigid_body_pos.clone()
        body_rot = self._rigid_body_rot.clone()
        body_dof_pos = self._dof_pos.clone()
        body_dof_vel = self._dof_vel.clone()

        motion_times = (self.progress_buf[env_ids] + 1) * self.dt
        reference_time = (self.progress_buf[env_ids] + 1 - self.reference_start_index) * self.dt
        reference_time = torch.clamp(reference_time, min=0, max=self.max_episode_length * self.dt)


        motion_res = self._get_mld_state_from_motionlib(env_ids, reference_time)
        time_steps = 1
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"]
  
        ## clone data from motion library
        ref_rb_pos = ref_rb_pos.clone()
        ref_rb_rot = ref_rb_rot.clone()
        ref_body_vel = ref_body_vel.clone()
        ref_traj_pos = self._traj_gen.calc_pos(env_ids,motion_times)
        ref_rb_pos[:, :, :2] -= ref_rb_pos[:, 0, :2].clone().unsqueeze(1)
        ref_rb_pos[:, :, 0:2] += ref_traj_pos[:, :2].unsqueeze(1)

        ## change heading via trajectory
        ## step-1 obatin trajectory
        ## step-2 compute relative heading
        ## step-3 change reference pos and rot
        trajectory = self.traj_samples.clone()
        traj_begin, traj_end = trajectory[:, 0], trajectory[:, 2]
        traj_heading = compute_traj_heading(traj_begin, traj_end)
        motion_heading = compute_motion_heading(ref_rb_rot[:, 0])
        ref_rb_pos, ref_rb_rot = process_imitation_target_with_heading(traj_heading, motion_heading, ref_rb_pos, ref_rb_rot)
        ref_root_rot = ref_rb_rot[:, 0, :].clone()
        ref_root_pos = ref_rb_pos[:, 0, :].clone()


        d3_visible = torch.zeros((env_ids.shape[0]), dtype=torch.int, device=self.device)
        mask = ((self.progress_buf[env_ids] + 1) >= self.reference_start_index[env_ids]) & ((self.progress_buf[env_ids] + 1) < self.reference_end_index[env_ids])
        d3_visible[mask] = 1

        if self._full_body_reward:
            body_pos = body_pos * self.tracking_mask
            body_rot = body_rot * self.tracking_mask
            body_dof_pos = body_dof_pos.reshape(self.num_envs, -1, 3) * self.tracking_mask[:, 1:]
            body_dof_pos = body_dof_pos.reshape(self.num_envs, -1)
            body_dof_vel = body_dof_vel.reshape(self.num_envs, -1, 3) * self.tracking_mask[:, 1:]
            body_dof_vel = body_dof_vel.reshape(self.num_envs, -1)
            ref_rb_pos = ref_rb_pos * self.tracking_mask
            ref_rb_rot = ref_rb_rot * self.tracking_mask
            dof_pos = dof_pos.reshape(self.num_envs, -1, 3) * self.tracking_mask[:, 1:]
            dof_pos = dof_pos.reshape(self.num_envs, -1)
            dof_vel = dof_vel.reshape(self.num_envs, -1, 3) * self.tracking_mask[:, 1:]
            dof_vel = dof_vel.reshape(self.num_envs, -1)

            im_reward, im_reward_raw = compute_imitation_reward(
                    root_pos, root_rot, body_pos, body_rot, body_dof_pos, body_dof_vel,
                    ref_rb_pos, ref_rb_rot, dof_pos, dof_vel,
                    self._dof_obs_size,
                    self._dof_offsets,
                    self.reward_specs, 
                    d3_visible)
        else:
            dof_tack_id = [i-1 for i in self._track_bodies_id[1:]]
            body_pos_subset = body_pos[..., self._track_bodies_id, :] * self.tracking_mask[:, self._track_bodies_id, :]
            body_rot_subset = body_rot[..., self._track_bodies_id, :] * self.tracking_mask[:, self._track_bodies_id, :]
            body_dof_pos_subset = body_dof_pos.reshape(self.num_envs, -1, 3)
            body_dof_pos_subset = body_dof_pos_subset[..., dof_tack_id, :] * self.tracking_mask[:, dof_tack_id, :]
            body_dof_pos_subset = body_dof_pos_subset.reshape(self.num_envs, -1)
            body_dof_vel_subset = body_dof_vel.reshape(self.num_envs, -1, 3)
            body_dof_vel_subset = body_dof_vel_subset[..., dof_tack_id, :] * self.tracking_mask[:, dof_tack_id, :]
            body_dof_vel_subset = body_dof_vel_subset.reshape(self.num_envs, -1)

            ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :] * self.tracking_mask[:, self._track_bodies_id, :]
            ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :] * self.tracking_mask[:, self._track_bodies_id, :]
            dof_pos_subset = dof_pos.reshape(self.num_envs, -1, 3)
            dof_pos_subset = dof_pos_subset[..., dof_tack_id, :] * self.tracking_mask[:, dof_tack_id, :]
            dof_pos_subset = dof_pos_subset.reshape(self.num_envs, -1)
            dof_vel_subset = dof_vel.reshape(self.num_envs, -1, 3)
            dof_vel_subset = dof_vel_subset[..., dof_tack_id, :] * self.tracking_mask[:, dof_tack_id, :]
            dof_vel_subset = dof_vel_subset.reshape(self.num_envs, -1)

            sub_dof_obs_size = 6 * len(self._track_bodies_id[1:])
            sub_dof_obs_offset = self._dof_offsets[:len(self._track_bodies_id[1:])+1]

            im_reward, im_reward_raw = compute_imitation_reward(
                    root_pos, root_rot, body_pos_subset, body_rot_subset,
                    body_dof_pos_subset, body_dof_vel_subset, ref_rb_pos_subset,
                    ref_rb_rot_subset, dof_pos_subset, dof_vel_subset,
                    sub_dof_obs_size,
                    sub_dof_obs_offset,
                    self.reward_specs,
                    d3_visible)
        return im_reward

    def _compute_reset(self):
        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs,
                               device=self.device,
                               dtype=torch.long)
        tar_pos = self._traj_gen.calc_pos(env_ids, time)
        ### ZL: entry point
        root_states = self._humanoid_root_states
        center_height = self.get_center_heights(
            root_states, env_ids=None).mean(dim=-1, keepdim=True)

        motion_times = (self.progress_buf[env_ids] + 1) * self.dt
        reference_time = (self.progress_buf[env_ids] + 1 - self.reference_start_index[env_ids]) * self.dt
        reference_time = torch.clamp(reference_time, min=0, max=self.max_episode_length * self.dt)


        motion_res = self._get_mld_state_from_motionlib(env_ids, reference_time)
        time_steps = 1
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"]

        d3_visible = torch.zeros((env_ids.shape[0]), dtype=torch.int, device=self.device)
        mask = ((self.progress_buf[env_ids] + 1) >= self.reference_start_index[env_ids]) & ((self.progress_buf[env_ids] + 1) < self.reference_end_index[env_ids])
        d3_visible[mask] = 1

        ## clone data from motion library
        ref_rb_pos = ref_rb_pos.clone()
        ref_rb_rot = ref_rb_rot.clone()
        ref_body_vel = ref_body_vel.clone()
        ref_traj_pos = self._traj_gen.calc_pos(env_ids,motion_times)
        ref_rb_pos[:, :, :2] -= ref_rb_pos[:, 0, :2].clone().unsqueeze(1)
        ref_rb_pos[:, :, 0:2] += ref_traj_pos[:, :2].unsqueeze(1)

        ## change heading via trajectory
        ## step-1 obatin trajectory
        ## step-2 compute relative heading
        ## step-3 change reference pos and rot
        trajectory = self.traj_samples.clone()
        traj_begin, traj_end = trajectory[:, 0], trajectory[:, 2]
        traj_heading = compute_traj_heading(traj_begin, traj_end)
        motion_heading = compute_motion_heading(ref_rb_rot[:, 0])
        ref_rb_pos, ref_rb_rot = process_imitation_target_with_heading(traj_heading, motion_heading, ref_rb_pos, ref_rb_rot)
        ref_root_rot = ref_rb_rot[:, 0, :].clone()
        ref_root_pos = ref_rb_pos[:, 0, :].clone()

        ## changed height
        root_state = torch.cat([ref_root_pos, ref_root_rot], dim=-1)
        center_heights = self.get_center_heights(root_states=root_state, env_ids=env_ids)
        center_heights = center_heights.mean(dim=-1, keepdim=True)
        ref_rb_pos[..., 2] += center_heights

        tracking_mask = self.tracking_mask[:, self._reset_body_ids]
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf, self.progress_buf, self._contact_forces,
            self._contact_body_ids, center_height, self._rigid_body_pos[:, self._reset_body_ids] ,
              ref_rb_pos[:, self._reset_body_ids], d3_visible, tracking_mask,
            tar_pos, self.max_episode_length, self._fail_dist,
            self._enable_early_termination, self._termination_heights, flags.no_collision_check,
            self.terminate_dist[0], self.use_imitation_reset)
        return



#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def heading_to_vec(h_theta):
    v = torch.stack([torch.cos(h_theta), torch.sin(h_theta)], dim=-1)
    return v

@torch.jit.script
def dof_to_obs(pose, dof_obs_size, dof_offsets):
    # type: (Tensor, int, List[int]) -> Tensor
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # jp hack
        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose)
        elif (dof_size == 1):
            axis = torch.tensor([0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device)
            joint_pose_q = quat_from_angle_axis(joint_pose[..., 0], axis)
        else:
            joint_pose_q = None
            assert(False), "Unsupported joint type"

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        dof_obs[:, (j * joint_obs_size):((j + 1) * joint_obs_size)] = joint_dof_obs

    assert((num_joints * joint_obs_size) == dof_obs_size)

    return dof_obs



#@torch.jit.script
def compute_imitation_observations(root_pos, root_rot, body_pos, body_rot, body_dof, ref_body_pos, ref_body_rot, ref_body_dof, d3_observation, tracking_mask, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,  Tensor, Tensor, Tensor, int, bool) -> Tensor
    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    ############ due to the real to sim assignment, we only use the xy obs for position

    ############# root relative state #############
    ############# diff root height #############
    # root_h = root_pos[:, 2:3]
    target_root_pos = ref_body_pos[:, 0, :]
    target_root_rot = ref_body_rot[:, 0, :]
    # diff_root_height = root_h - target_root_pos[:, 2:3]

    ############# rel root ref obs #############
    heading_rot, heading = torch_utils.calc_heading_quat_inv_with_heading(root_rot)
    target_heading_rot, target_heading = torch_utils.calc_heading_quat_inv_with_heading(target_root_rot)
    target_rel_root_rot = quat_mul(target_root_rot, quat_conjugate(root_rot))
    if (d3_observation==0).sum() > 0:
        target_rel_root_rot[d3_observation==0] = torch.tensor([0, 0, 0, 1]).reshape(1, 4).repeat((d3_observation==0).sum(), 1).float().to(root_rot.device)
    target_rel_root_rot_obs = torch_utils.quat_to_tan_norm(target_rel_root_rot)

    ############# rel 2d pos #############
    target_rel_pos = target_root_pos[:, :3] - root_pos[:, :3]
    target_rel_pos = torch_utils.my_quat_rotate(heading_rot, target_rel_pos)
    target_rel_2d_pos = target_rel_pos[:, :2]

    ############# diff root heading #############
    target_rel_heading = target_heading - heading
    target_rel_heading_vec = heading_to_vec(target_rel_heading)

    ############# diff dof #############
    target_rel_dof_pos = ref_body_dof - body_dof

    ############# diff body pos #############
    target_rel_root_body_pos = ref_body_pos - target_root_pos.view(B, 1, 3)
    rel_root_body_pos = body_pos - root_pos.view(B, 1, 3)
    target_rel_pos = target_rel_root_body_pos - rel_root_body_pos
    target_rel_pos *= tracking_mask
    num_joints = target_rel_pos.shape[1]
    target_rel_pos = target_rel_pos.reshape(-1, 3)
    target_rel_pos = torch_utils.my_quat_rotate(heading_rot[:, None, :].repeat(1, J, 1).reshape(-1, 4), target_rel_pos)
    target_rel_pos = target_rel_pos.reshape(B, -1, 3)
    target_rel_pos = target_rel_pos[:, 1:]

    ############# whole body relative state #############
    #heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat( (1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_global_body_pos[..., 2] *= 0
    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(body_rot).repeat_interleave(time_steps, 0).view(B, time_steps, J, 4))
    diff_local_body_pos = torch_utils.my_quat_rotate(heading_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3)).reshape(B, time_steps, J, 3) * tracking_mask.unsqueeze(1)

    if (d3_observation==0).sum() >0:
        diff_local_body_pos[d3_observation==0] *= 0
        target_rel_2d_pos[d3_observation==0] *= 0
        target_rel_dof_pos[d3_observation==0] *= 0
        target_rel_pos[d3_observation==0] *= 0
        target_rel_heading_vec_tmp = target_rel_heading_vec[d3_observation==0]
        target_rel_heading_vec_tmp[:, 0] = 1
        target_rel_heading_vec_tmp[:, 1] *= 0
        target_rel_heading_vec[d3_observation==0] = target_rel_heading_vec_tmp
        diff_global_body_rot[d3_observation==0] = torch.tensor([0, 0, 0, 1]).reshape(1, 1, 1, 4).repeat((d3_observation==0).sum(), time_steps, J, 1).float().to(body_rot.device)

    diff_local_body_pos_flat = diff_local_body_pos.view(-1, 3)[:, :2].contiguous()
    diff_local_body_rot_flat =  diff_global_body_rot.view(-1, 4)

    obs.append(target_rel_root_rot_obs.view(B, -1))
    obs.append(target_rel_2d_pos.view(B, -1))
    obs.append(target_rel_pos.view(B, -1))
    obs.append(target_rel_heading_vec.view(B, -1))
    obs.append(target_rel_dof_pos.view(B, -1))
    obs.append(diff_local_body_pos_flat.view(B, -1)) # 1 * 10 * 3 * 3
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, -1)) #  1 * 10 * 3 * 6


    obs = torch.cat(obs, dim=-1)
    return obs

@torch.jit.script
def compute_imitation_reward(root_pos, root_rot, body_pos, body_rot, dof_pos, dof_vel, 
    ref_body_pos, ref_body_rot,  ref_dof_pos, ref_dof_vel, dof_obs_size, dof_offsets, rwd_specs, d3_visible):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int,  List[int], Dict[str, float], Tensor) -> Tuple[Tensor, Tensor]
    k_pos, k_rot, k_vel, k_dof = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"], rwd_specs["k_dof"]
    w_pos, w_rot, w_vel, w_dof = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"], rwd_specs["w_dof"]

    # dof rot reward
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    target_dof_obs = dof_to_obs(ref_dof_pos, dof_obs_size, dof_offsets)
    diff_dof_obs = dof_obs - target_dof_obs
    diff_dof_obs_dist = (diff_dof_obs ** 2).mean(dim=-1) * d3_visible
    r_dof = torch.exp(-k_dof * diff_dof_obs_dist) 

    # body position reward
    diff_global_body_pos = (ref_body_pos - ref_body_pos[:, :1]) - (body_pos - root_pos.unsqueeze(1))
    diff_body_pos_dist = (diff_global_body_pos[:, 1:] ** 2).mean(dim = -1).mean(dim = -1)  ######## use root relative position, path reward is computed in trajectory following
    r_body_pos = torch.exp(-k_pos * diff_body_pos_dist) 

    # body rotation reward
    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot, torch_utils.quat_conjugate(body_rot))
    diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
    diff_global_body_angle_dist = (diff_global_body_angle ** 2).mean(dim=-1) * d3_visible
    r_body_rot = torch.exp(-k_rot * diff_global_body_angle_dist)

    # velocity reward
    diff_dof_vel = ref_dof_vel - dof_vel
    diff_dof_vel_dist = (diff_dof_vel  ** 2).mean(dim=-1) * d3_visible
    r_dof_vel = torch.exp(-k_vel * diff_dof_vel_dist)

    reward = w_pos * r_body_pos + w_rot * r_body_rot + w_vel * r_dof_vel + r_dof * w_dof
    reward_raw = torch.stack([r_body_pos, r_body_rot, r_dof_vel, r_dof], dim = -1)
    return reward, reward_raw


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf,
                           contact_body_ids, center_height, rigid_body_pos, ref_body_pos, d3_visible, tracking_mask,
                           tar_pos, max_episode_length, fail_dist,
                           enable_early_termination, termination_heights, disableCollision, termination_distance, use_imitation_reset):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, bool, Tensor, bool, float, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        force_threshold = 50
        body_contact_force = torch.sqrt(torch.square(torch.abs(masked_contact_buf.sum(dim=-2))).sum(dim=-1)) > force_threshold
        has_fallen = body_contact_force
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)

        root_pos = rigid_body_pos[..., 0, :]
        tar_delta = tar_pos[..., 0:2] - root_pos[...,0:2]  # also reset if toooo far away from the target trajectory
        tar_dist_sq = torch.sum(tar_delta * tar_delta, dim=-1)
        tar_fail = tar_dist_sq > fail_dist * fail_dist

        has_failed = torch.logical_or(has_fallen, tar_fail)


        ####### imitation target


        if disableCollision:
            has_failed[:] = False

        if use_imitation_reset:
            rigid_body_pos = (rigid_body_pos - rigid_body_pos[...,0:1,:]) * tracking_mask
            ref_body_pos = (ref_body_pos - ref_body_pos[...,0:1,:]) * tracking_mask
            imitation_fallen = torch.any(torch.norm(rigid_body_pos[:, 1:] - ref_body_pos[:, 1:], dim=-1) > termination_distance, dim = -1) * d3_visible
            has_failed = torch.logical_or(has_failed, imitation_fallen)

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset_buf), terminated)

    return reset, terminated


#@torch.jit.script
def process_imitation_target_with_heading(trajectory_heading_matrix, motion_heading_matrix, ref_body_pos, ref_body_rot):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    relative_heading_rotation = torch.matmul(trajectory_heading_matrix, torch.transpose(motion_heading_matrix, -1, -2))

    ### change position orientation
    ref_body_root = ref_body_pos[:, 0:1, :]
    relative_body_pos = ref_body_pos - ref_body_root
    relative_body_pos = torch.matmul(relative_heading_rotation, relative_body_pos.permute(0, 2, 1)).permute(0, 2, 1)
    ref_bdoy_pos = relative_body_pos + ref_body_root

    ### change global rotation
    ref_body_rot = konia_transform.quaternion_to_rotation_matrix(ref_body_rot[:, :, [3, 0, 1, 2]])
    ref_body_rot = torch.matmul(relative_heading_rotation.unsqueeze(1), ref_body_rot)
    ref_body_rot = konia_transform.rotation_matrix_to_quaternion(ref_body_rot)[:, :, [1, 2, 3, 0]]
    return ref_bdoy_pos, ref_body_rot
    
#@torch.jit.script
def compute_traj_heading(traj_beign, traj_end):
    # type: (Tensor, Tensor) -> Tensor
    traj_direction = traj_end - traj_beign
    traj_norm = torch.norm(traj_direction, dim=-1)
    traj_direction = traj_direction / (1e-5 + traj_norm.unsqueeze(-1))
    traj_heading = torch.atan2(traj_direction[..., 1], traj_direction[..., 0])
    traj_heading_rotation = angle_to_rotation_matrix(traj_heading)
    return traj_heading_rotation

#@torch.jit.script
def compute_motion_heading(motion_rotation):
    #type: (Tensor) -> Tensor
    motion_heading = torch_utils.calc_heading(motion_rotation)
    motion_heading_rotation = angle_to_rotation_matrix(motion_heading)

    return motion_heading_rotation

#@torch.jit.script
def angle_to_rotation_matrix(angle):
    rotation_matrix = torch.zeros((angle.shape[0], 3, 3), dtype=torch.float32, device=angle.device)
    rotation_matrix[:, 0, 0] = torch.cos(angle)
    rotation_matrix[:, 0, 1] = -torch.sin(angle)
    rotation_matrix[:, 1, 0] = torch.sin(angle)
    rotation_matrix[:, 1, 1] = torch.cos(angle)
    rotation_matrix[:, 2, 2] = 1
    return rotation_matrix