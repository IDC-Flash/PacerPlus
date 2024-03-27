# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from shutil import ExecError
import torch
import numpy as np
import env.util.traj_generator as traj_generator
import joblib
import amp.env.tasks.humanoid_traj as humanoid_traj

from amp.utils.flags import flags
from amp.utils.motion_lib import MotionLib
from amp.utils.motion_lib_h1 import MotionLib as MotionLibH1
from amp.env.tasks.humanoid_traj import compute_location_reward
from amp.utils.konia_transform import quaternion_to_angle_axis

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from env.tasks.humanoid import dof_to_obs
from env.tasks.humanoid_amp import  remove_base_rot
from utils import torch_utils
from poselib.poselib.core.rotation3d import  quat_mul
from scipy.spatial.transform import Rotation as sRot
from uhc.smpllib.smpl_mujoco import SMPL_BONE_ORDER_NAMES as joint_names


HACK_MOTION_SYNC = False

class HumanoidPedestrianIm(humanoid_traj.HumanoidTraj):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._full_body_reward = cfg["env"].get("full_body_reward", True)
        self._min_motion_len = cfg["env"].get("min_length", -1)

        self.different_motion_file = cfg["env"].get("different_motion_file", True)
        self.reset_buffer = cfg["env"].get("reset_buffer", 0)
        self._track_bodies_id = [i for i in range(12, 22)]
        self._dof_track_bodies_id = [i for i in range(11, 15)] + [i for i in range(16, 20)]
        #### input tracking mask into the observation
        if not cfg['args'].headless:
            self._infilling_handles = [[] for _ in range(cfg['args'].num_envs)]

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        self.reward_raw = torch.zeros((self.num_envs, 4 + self.num_locomotion_reward)).to(self.device)
        self.terminate_dist = cfg['env'].get('terminate_dist', 0.4), 
        self.use_imitation_reset = cfg['env'].get('use_imitation_reset', False)

        self.reward_specs = cfg["env"].get(
            "reward_specs", {
                "k_pos": 20,
                "k_vel": 0.02,
                "k_dof": 5,
                "w_pos": 0.2,
                "w_vel": 0.1,
                "w_dof": 0.7,
            })
        print("Reward specs: ", self.reward_specs)
        self.task_reward_specs = cfg["env"].get(
            "task_reward_specs", {
                "w_location": 0.7,
                "w_imitation": 0.3,
            }
        )



        self.imitation_ref_motion_cache = {}
        self.d3_visible = torch.zeros((self.num_envs), dtype=torch.int, device=self.device)
        self.im_ref_rb_target_pos = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device).float()
        self.target_dof = torch.zeros((self.num_envs, self.num_dofs), device=self.device).float()


        if not self.headless:
            self._build_infilling_marker_state_tensors()

    def post_physics_step(self):
        super().post_physics_step()
        self._set_target_motion_state()


    def _build_env(self, env_id, env_ptr, humanoid_asset, dof_props_asset, rigid_shape_props_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset, dof_props_asset, rigid_shape_props_asset)
        if not self.headless:
            self._build_infilling_marker(env_id, env_ptr)

    def _draw_task(self):
        self._update_marker()
        return
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 2 * self._num_traj_samples
            obs_size += 38 + 1
        return obs_size
    
    def get_task_obs_size_detail(self):
        task_obs_detail = []
        if (self._enable_task_obs):
            task_obs_detail.append(["traj", 2 * self._num_traj_samples])
        task_obs_detail.append(["imitation_target", 42])
        task_obs_detail.append(["imitation_target_visible", 1])
        return task_obs_detail

    def _update_marker(self):
        env_ids = torch.arange(self.num_envs).to(self.device)
        self._infilling_pos[:] = self.im_ref_rb_target_pos[:, self._track_bodies_id].clone() + self._humanoid_root_states[..., 0:3].unsqueeze(1)
        traj_samples = self._fetch_traj_samples()
        self._marker_pos[:] = traj_samples
        self._marker_pos[..., 2] = self._humanoid_root_states[..., 2:3]  # jp hack # ZL hack
        
        ref_start = torch.tensor(self.reference_start_index).to(self.device)
        ref_length = torch.tensor(self.reference_length).to(self.device)
        d3_visible = torch.zeros((env_ids.shape[0]), dtype=torch.int, device=self.device)
        mask = ((self.progress_buf[env_ids] + 1) >= ref_start[env_ids]) & ((self.progress_buf[env_ids] + 1) < ref_start[env_ids] + ref_length[env_ids])
        d3_visible[mask] = 1
        
        if d3_visible.sum() > 0:
            comb_idx = torch.cat([self._traj_marker_actor_ids, self._infilling_actor_ids])
        else:
            comb_idx = self._traj_marker_actor_ids

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(comb_idx),
                                                     len(comb_idx))

        return
    
    def _build_infilling_marker(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        for i in range(len(self._track_bodies_id)):
            marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", self.num_envs + 5, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.8, 0.0))
            self._infilling_handles[env_id].append(marker_handle)

        return

    def _build_infilling_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._infilling_states = self._root_states.view(
            self.num_envs, num_actors,
            self._root_states.shape[-1])[..., 11:(11 + len(self._track_bodies_id)), :]
        self._infilling_pos = self._infilling_states[..., :3]
        self._infilling_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._infilling_handles, dtype=torch.int32, device=self.device)
        self._infilling_actor_ids = self._infilling_actor_ids.flatten()
        return


    def _load_motion(self, motion_file):
        self._motion_lib = MotionLibH1( motion_file=motion_file, fps = 60, device=self.device)
        self._motion_lib.load_motions(num_envs=self.num_envs)

        if self.different_motion_file:
            self._imitation_motion_lib = MotionLibH1( motion_file=self.cfg['env']['imitation_motion_file'], fps = 60, device=self.device)
            self._imitation_motion_lib.load_motions(num_envs=self.num_envs)
        else:
            self._imitation_motion_lib = self._motion_lib


        self.motion_steps = (self._imitation_motion_lib._motion_lengths / self.dt).int()
        self.reference_start_index = torch.zeros((self.num_envs), dtype=torch.int, device=self.device)
        self.reference_length = torch.clamp(self.motion_steps, self.max_episode_length // 2)
        return

    def resample_motions(self):
        print("Partial solution, only resample motions...")
        self._motion_lib.load_motions(self.num_envs) # For now, only need to sample motions since there are only 400 hmanoids
        if hasattr(self, '_imitation_motion_lib') and self.different_motion_file:
            self._imitation_motion_lib.load_motions(self.num_envs)

    def _get_smpl_state_from_imitation_motionlib_cache(self, motion_ids, motion_times):
        motion_times = torch.clamp(motion_times, 0)
        motion_res = self._imitation_motion_lib.get_motion_state(motion_ids, motion_times)    
        if self.has_hand_dof:
            dof_pos = torch.zeros(motion_res["dof_pos"].shape[0], motion_res["dof_pos"].shape[1]+2, device=self.device)
            dof_pos[:, self.amp_dof_idx] = motion_res["dof_pos"]
            motion_res["dof_pos"] = dof_pos

            dof_vel = torch.zeros(motion_res["dof_vel"].shape[0],motion_res["dof_vel"].shape[1]+2, device=self.device)
            dof_vel[:, self.amp_dof_idx] = motion_res["dof_vel"]
            motion_res["dof_vel"] = dof_vel    
        return motion_res
    

    def _reset_task(self, env_ids):
        root_pos = self._humanoid_root_states[env_ids, 0:3]
        motion_steps = (self._imitation_motion_lib._motion_lengths / self.dt).int()
        self.reference_start_index[env_ids] = torch.zeros((env_ids.shape[0]), dtype=torch.int, device=self.device)
        self.reference_length[env_ids] = torch.clamp(motion_steps[env_ids], self.max_episode_length // 2)
        self._traj_gen.reset(env_ids, root_pos)
        return


    def _reset_ref_state_init(self, env_ids):
        super()._reset_ref_state_init(env_ids)
        use_env_ids = not (len(env_ids) == self.num_envs and torch.all(env_ids == torch.arange(self.num_envs, device=self.device)))
        if not use_env_ids:
            self._set_target_motion_state()
        else:
            self._set_target_motion_state(env_ids)
        return
    
    def _sample_ref_state(self, env_ids, vel_min=1, vel_range=0.5):
        num_envs = env_ids.shape[0]
        motion_ids = env_ids
        motion_times = torch.tensor(self.reference_start_index).to(self.device)[env_ids] * self.dt


        motion_res = self._get_smpl_state_from_imitation_motionlib_cache(motion_ids, motion_times)


        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, local_pos = \
              motion_res["root_pos"], motion_res["root_rot"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_pos"], motion_res["dof_vel"], motion_res["key_pos"], motion_res["local_pos"]



        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, local_pos
    
    def _set_target_motion_state(self, env_ids=None):
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long) if env_ids is None else env_ids

        ref_start = self.reference_start_index
        ref_length = self.reference_length
        motion_times = (self.progress_buf[env_ids] + 1) * self.dt

        motion_res = self._get_smpl_state_from_imitation_motionlib_cache(env_ids, motion_times)
        time_steps = 1

        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, local_pos = \
              motion_res["root_pos"], motion_res["root_rot"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_pos"], motion_res["dof_vel"], motion_res["key_pos"], motion_res["local_pos"]
        
        if flags.test:
            d3_visible = torch.zeros((env_ids.shape[0]), dtype=torch.int, device=self.device)
            mask = ((self.progress_buf[env_ids] + 1) >= ref_start[env_ids]) & ((self.progress_buf[env_ids] + 1) < ref_start[env_ids] + ref_length[env_ids])
            d3_visible[mask] = 1
            self.d3_visible[env_ids] = d3_visible
            self.target_dof[env_ids] = dof_pos.clone()

    # def pre_physics_step(self, actions):
    #     #### Hz < 500 use PD control rather than torque control
    #     pd_tar = self._pd_action_offset + self._pd_action_scale * actions 
    #     pd_tar = torch.clamp(pd_tar, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
    #     if self.d3_visible.sum() > 0:
    #         pd_tar[self.d3_visible==1, self._dof_track_bodies_id] = self.target_dof[self.d3_visible==1, self._dof_track_bodies_id]

    #     pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
    #     self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
    #     return

    def _compute_task_obs(self, env_ids=None):
        # Compute task observations (terrain, trajectory, self state)
        basic_obs = super()._compute_task_obs(env_ids)
        # Compute IM observations
        if (env_ids is None):
            body_pos = self._rigid_body_pos.clone()
            body_rot = self._rigid_body_rot.clone()
            body_dof = self.dof_pos.clone()
            env_ids = torch.arange(self.num_envs,
                               dtype=torch.long,
                               device=self.device)
        else:
            body_pos = self._rigid_body_pos[env_ids].clone()
            body_rot = self._rigid_body_rot[env_ids].clone()
            body_dof = self.dof_pos[env_ids].clone()

        
        ######### we need a flag for this observation
        ######### if flag == 1, we need to compute the observation
        ######### if flag == 0, we need do not need the imitation target in the observation

        ref_start = self.reference_start_index
        ref_length = self.reference_length
        motion_times = (self.progress_buf[env_ids] + 1) * self.dt

        motion_res = self._get_smpl_state_from_imitation_motionlib_cache(env_ids, motion_times)
        time_steps = 1
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, local_pos = \
              motion_res["root_pos"], motion_res["root_rot"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_pos"], motion_res["dof_vel"], motion_res["key_pos"], motion_res["local_pos"]

        d3_visible = torch.zeros((env_ids.shape[0]), dtype=torch.int, device=self.device)
        mask = ((self.progress_buf[env_ids] + 1) >= ref_start[env_ids]) & ((self.progress_buf[env_ids] + 1) < ref_start[env_ids] + ref_length[env_ids])
        d3_visible[mask] = 1

        ref_root_pos = root_pos.clone()
        ref_root_rot = root_rot.clone()
        ref_rb_pos = local_pos.clone()
        ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :] 
        ref_dof_pos_subset = dof_pos[..., self._dof_track_bodies_id]

        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]
        body_pos_subset = body_pos[..., self._track_bodies_id, :] 
        body_dof_subset = body_dof[..., self._dof_track_bodies_id] 


        obs = compute_imitation_observations(root_pos, root_rot, body_pos_subset, body_dof_subset,
                                            ref_root_pos, ref_root_rot, ref_rb_pos_subset, ref_dof_pos_subset,
                                            d3_visible,  time_steps, True)
        
        obs = torch.cat([basic_obs, obs, d3_visible.unsqueeze(-1)], dim=-1)


        self.d3_visible[env_ids] = d3_visible

        ########### update for visualization
        if flags.test:  
            # rotate
            ref_heading_rot, heading = torch_utils.calc_heading_quat_inv_with_heading(ref_root_rot)
            B, J, _ = body_pos.shape
            ref_body_pos = ref_rb_pos.clone()
            ref_body_pos = ref_body_pos.reshape(-1, 3)
            # rot = torch.tensor([ 0, 0, -0.7071068, 0.7071068 ], device=self.device, dtype=torch.float32)[None, None, :]
            # ref_body_pos = torch_utils.my_quat_rotate(rot.repeat(1, J, 1).reshape(-1, 4), ref_body_pos)
            ref_body_pos = torch_utils.my_quat_rotate(ref_heading_rot[:, None, :].repeat(1, J, 1).reshape(-1, 4), ref_body_pos)
            heading = torch_utils.calc_heading_quat(root_rot)
            ref_body_pos = torch_utils.my_quat_rotate(heading[:, None, :].repeat(1, J, 1).reshape(-1, 4), ref_body_pos)
            ref_body_pos = ref_body_pos.reshape(B, -1, 3)
            self.im_ref_rb_target_pos[env_ids] = ref_body_pos.clone()
            self.im_ref_rb_target_pos[:, 0] *= -1
        return obs


    def _compute_reward(self, actions):
        w_location, w_imitation = self.task_reward_specs['w_location'], self.task_reward_specs['w_imitation']

        root_pos = self._humanoid_root_states[..., 0:3]

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self._traj_gen.calc_pos(env_ids, time)

        location_reward = compute_location_reward(root_pos, tar_pos)
        im_reward, im_reward_raw = self._compute_im_rewards(actions)
        locomotion_reward = self._build_locomotion_rewards()

        self.rew_buf[:] = location_reward * w_location + im_reward * w_imitation + locomotion_reward.sum(dim=-1)
        self.reward_raw[:] = torch.cat([location_reward[:, None], im_reward_raw, locomotion_reward], dim = -1)
        return


    def _compute_im_rewards(self, actions):

        env_ids = torch.arange(self.num_envs,
                               dtype=torch.long,
                               device=self.device)
        
        body_pos = self._rigid_body_pos.clone()
        body_rot = self._rigid_body_rot.clone()
        body_dof_pos = self.dof_pos.clone()
        body_dof_vel = self.dof_vel.clone()

        ref_start = self.reference_start_index
        ref_length = self.reference_length
        motion_times = (self.progress_buf[env_ids]) * self.dt

        motion_res = self._get_smpl_state_from_imitation_motionlib_cache(env_ids, motion_times)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, local_pos = \
              motion_res["root_pos"], motion_res["root_rot"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_pos"], motion_res["dof_vel"], motion_res["key_pos"], motion_res["local_pos"]

        ### prepare for reward

        d3_visible = torch.zeros((self.num_envs), dtype=torch.int, device=self.device)
        mask = ((self.progress_buf[env_ids]) >= ref_start[env_ids]) & ((self.progress_buf[env_ids]) < ref_start[env_ids] + ref_length[env_ids])
        d3_visible[mask] = 1

        
        ref_root_pos = root_pos.clone()
        ref_root_rot = root_rot.clone()
        ref_rb_pos = local_pos[:, self._track_bodies_id]
        ref_dof_pos = dof_pos[:, self._dof_track_bodies_id]
        ref_dof_vel = dof_vel[:, self._dof_track_bodies_id]

        root_pos = body_pos[:, 0]
        root_rot = body_rot[:, 0]
        body_pos = body_pos[:, self._track_bodies_id] 
        body_dof_pos = body_dof_pos[:, self._dof_track_bodies_id]
        body_dof_vel = body_dof_vel[:, self._dof_track_bodies_id]

        im_reward, im_reward_raw = compute_imitation_reward(
                root_pos, root_rot, body_pos,  body_dof_pos, body_dof_vel,
                ref_root_pos, ref_root_rot, ref_rb_pos, ref_dof_pos, ref_dof_vel,
                self.reward_specs, 
                d3_visible)

        return im_reward, im_reward_raw

    def _compute_reset(self):
        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs,
                               device=self.device,
                               dtype=torch.long)
        tar_pos = self._traj_gen.calc_pos(env_ids, time)
        ref_start = self.reference_start_index
        ref_length = self.reference_length
        motion_times = (self.progress_buf[env_ids]) * self.dt

        motion_res = self._get_smpl_state_from_imitation_motionlib_cache(env_ids, motion_times)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, local_pos = \
              motion_res["root_pos"], motion_res["root_rot"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_pos"], motion_res["dof_vel"], motion_res["key_pos"], motion_res["local_pos"]


        d3_visible = torch.zeros((self.num_envs), dtype=torch.int, device=self.device)
        mask = ((self.progress_buf[env_ids]) >= ref_start[env_ids] + self.reset_buffer) & ((self.progress_buf[env_ids]) < ref_start[env_ids] + ref_length[env_ids])
        d3_visible[mask] = 1

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf, self.progress_buf, self._contact_forces, self._termination_contact_body_ids,
            self._rigid_body_pos[:,0], self._rigid_body_rot[:, 0], self._rigid_body_pos[:, self._track_bodies_id],
            root_pos, root_rot,  local_pos[:, self._track_bodies_id], 
            d3_visible, tar_pos, self.max_episode_length, self._fail_dist,
            self._enable_early_termination, flags.no_collision_check,
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

#@torch.jit.script
def compute_imitation_observations(root_pos, root_rot, body_pos,  body_dof, 
                                   ref_root_pos, ref_root_rot, ref_body_pos,
                                     ref_body_dof, d3_observation, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, bool) -> Tensor
    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    ############ due to the real to sim assignment, we only use the xy obs for position

    ############# root relative state #############
    ############# diff root height #############
    # root_h = root_pos[:, 2:3]
    target_root_pos = ref_root_pos
    target_root_rot = ref_root_rot
    # diff_root_height = root_h - target_root_pos[:, 2:3]

    ############# rel root ref obs #############
    heading_rot, heading = torch_utils.calc_heading_quat_inv_with_heading(root_rot)
    target_heading_rot, target_heading = torch_utils.calc_heading_quat_inv_with_heading(target_root_rot)

    ############# diff dof #############
    target_rel_dof_pos = ref_body_dof - body_dof

    ############# diff body pos #############
    target_rel_root_body_pos = ref_body_pos - target_root_pos.view(B, 1, 3)
    rel_root_body_pos = body_pos - root_pos.view(B, 1, 3)
    target_rel_root_body_pos = target_rel_root_body_pos.reshape(-1, 3)
    target_rel_root_body_pos = torch_utils.my_quat_rotate(target_heading_rot[:, None, :].repeat(1, J, 1).reshape(-1, 4), target_rel_root_body_pos)
    target_rel_root_body_pos = target_rel_root_body_pos.reshape(B, -1, 3)

    rel_root_body_pos = rel_root_body_pos.reshape(-1, 3)
    rel_root_body_pos = torch_utils.my_quat_rotate(heading_rot[:, None, :].repeat(1, J, 1).reshape(-1, 4), rel_root_body_pos)
    rel_root_body_pos = rel_root_body_pos.reshape(B, -1, 3)

    target_rel_pos = target_rel_root_body_pos - rel_root_body_pos


    if (d3_observation==0).sum() >0:
        target_rel_dof_pos[d3_observation==0] *= 0
        target_rel_pos[d3_observation==0] *= 0

    obs.append(target_rel_pos.view(B, -1))
    obs.append(target_rel_dof_pos.view(B, -1))
    obs = torch.cat(obs, dim=-1)
    return obs

#@torch.jit.script
def compute_imitation_reward(root_pos, root_rot, body_pos, dof_pos, dof_vel, 
    ref_root_pos, ref_root_rot, ref_body_pos, ref_dof_pos, ref_dof_vel, rwd_specs, d3_visible):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], Tensor) -> Tuple[Tensor, Tensor]
    k_pos, k_vel, k_dof = rwd_specs["k_pos"], rwd_specs["k_vel"], rwd_specs["k_dof"]
    w_pos, w_vel, w_dof = rwd_specs["w_pos"],  rwd_specs["w_vel"], rwd_specs["w_dof"]

    # dof rot reward
    diff_dof_obs = dof_pos - ref_dof_pos
    diff_dof_obs_dist = (diff_dof_obs ** 2).mean(dim=-1) * d3_visible
    r_dof = torch.exp(-k_dof * diff_dof_obs_dist) 

    # body position reward
    heading_rot, heading = torch_utils.calc_heading_quat_inv_with_heading(root_rot)
    target_heading_rot, target_heading = torch_utils.calc_heading_quat_inv_with_heading(ref_root_rot)
    B, J, _ = body_pos.shape

    target_rel_root_body_pos = ref_body_pos - ref_root_pos.view(B, 1, 3)
    rel_root_body_pos = body_pos - root_pos.view(B, 1, 3)
    target_rel_root_body_pos = target_rel_root_body_pos.reshape(-1, 3)
    target_rel_root_body_pos = torch_utils.my_quat_rotate(target_heading_rot[:, None, :].repeat(1, J, 1).reshape(-1, 4), target_rel_root_body_pos)
    target_rel_root_body_pos = target_rel_root_body_pos.reshape(B, -1, 3)

    rel_root_body_pos = rel_root_body_pos.reshape(-1, 3)
    rel_root_body_pos = torch_utils.my_quat_rotate(heading_rot[:, None, :].repeat(1, J, 1).reshape(-1, 4), rel_root_body_pos)
    rel_root_body_pos = rel_root_body_pos.reshape(B, -1, 3)
    diff_body_pos = rel_root_body_pos - target_rel_root_body_pos
    diff_body_pos_dist = (diff_body_pos ** 2).mean(dim = -1).mean(dim = -1) * d3_visible
     ######## use root relative position, path reward is computed in trajectory following
    r_body_pos = torch.exp(-k_pos * diff_body_pos_dist) 


    # velocity reward
    diff_dof_vel = ref_dof_vel - dof_vel
    diff_dof_vel_dist = (diff_dof_vel  ** 2).mean(dim=-1) * d3_visible
    r_dof_vel = torch.exp(-k_vel * diff_dof_vel_dist)

    reward = w_pos * r_body_pos + w_vel * r_dof_vel + r_dof * w_dof
    reward_raw = torch.stack([r_body_pos, r_dof_vel, r_dof], dim = -1)
    return reward, reward_raw


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf,
                           termination_contact_body_ids, 
                            root_pos, root_rot, rigid_body_pos,
                            ref_root_pos, ref_root_rot, ref_body_pos, d3_visible,
                           tar_pos, max_episode_length, fail_dist,
                           enable_early_termination,  disableCollision, termination_distance, use_imitation_reset):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, bool,  bool, float, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        fall_contact = torch.any(torch.norm(contact_buf[:,termination_contact_body_ids], dim=-1) > 1, dim=1)


        rpy = get_euler_xyz(root_rot)
        fall_rpy = torch.logical_or(torch.abs(rpy[:,1])>1.0, torch.abs(rpy[:,0])>0.8)

        tar_delta = tar_pos[..., 0:2] - root_pos[..., 0:2]
        tar_dist_sq = torch.sum(tar_delta * tar_delta, dim=-1)
        tar_fail = tar_dist_sq > fail_dist * fail_dist

        has_fallen = torch.logical_or(fall_contact, fall_rpy)
        has_fallen = torch.logical_or(has_fallen, tar_fail)


        ####### imitation target
        if disableCollision:
            has_fallen[:] = False

        if use_imitation_reset:
            B, J, _ = rigid_body_pos.shape
            heading_rot, heading = torch_utils.calc_heading_quat_inv_with_heading(root_rot)
            target_heading_rot, target_heading = torch_utils.calc_heading_quat_inv_with_heading(ref_root_rot)
            target_rel_root_body_pos = ref_body_pos - ref_root_pos.view(B, 1, 3)
            rel_root_body_pos = rigid_body_pos - root_pos.view(B, 1, 3)

            target_rel_root_body_pos = target_rel_root_body_pos.reshape(-1, 3)
            target_rel_root_body_pos = torch_utils.my_quat_rotate(target_heading_rot[:, None, :].repeat(1, J, 1).reshape(-1, 4), target_rel_root_body_pos)
            target_rel_root_body_pos = target_rel_root_body_pos.reshape(B, -1, 3)

            rel_root_body_pos = rel_root_body_pos.reshape(-1, 3)
            rel_root_body_pos = torch_utils.my_quat_rotate(heading_rot[:, None, :].repeat(1, J, 1).reshape(-1, 4), rel_root_body_pos)
            rel_root_body_pos = rel_root_body_pos.reshape(B, -1, 3)

            imitation_fallen = torch.any(torch.norm(target_rel_root_body_pos - rel_root_body_pos, dim=-1) > termination_distance, dim = -1) * d3_visible
            has_failed = torch.logical_or(has_fallen, imitation_fallen)

        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset_buf), terminated)

    return reset, terminated