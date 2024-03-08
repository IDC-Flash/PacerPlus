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
import env.tasks.humanoid_pedestrain_terrain as humanoid_pedestrain_terrain

from amp.utils.flags import flags
from amp.utils.motion_lib import MotionLib
from amp.utils.motion_lib_smpl import MotionLib as MotionLibSMPL
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

class HumanoidPedestrianTerrainIm(humanoid_pedestrain_terrain.HumanoidPedestrianTerrain):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._full_body_reward = cfg["env"].get("full_body_reward", True)
        self._min_motion_len = cfg["env"].get("min_length", -1)
        self.use_different_motion_file = cfg["env"].get("use_different_motion_file", True)
        self.reset_buffer = cfg["env"].get("reset_buffer", 0)

        #### input tracking mask into the observation
        self.has_tracking_mask = cfg["env"].get("has_tracking_mask", False)
        self.has_tracking_mask_obs = cfg["env"].get("has_tracking_mask_obs", False) and self.has_tracking_mask
        if not cfg['args'].headless:
            self._infilling_handles = [[] for _ in range(cfg['args'].num_envs)]

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

        self.reward_raw = torch.zeros((self.num_envs, 3)).to(self.device)
        self.terminate_dist = cfg['env'].get('terminate_dist', 0.4), 
        self.use_imitation_reset = cfg['env'].get('use_imitation_reset', False)

        self.reward_specs = cfg["env"].get(
            "reward_specs", {
                "k_pos": 100,
                "k_rot": 10,
                "k_vel": 0.2,
                "k_dof": 60,
                "w_pos": 0.6,
                "w_rot": 0.2,
                "w_vel": 0.1,
                "w_dof": 0.1,
            })
        print("Reward specs: ", self.reward_specs)
        self.task_reward_specs = cfg["env"].get(
            "task_reward_specs", {
                "w_location": 0.5,
                "w_imitation": 0.5,
            }
        )

        if not self.headless:
            self._build_infilling_marker_state_tensors()

        self.imitation_ref_motion_cache = {}
        self.im_dof_pos = torch.zeros((self.num_envs, self.num_dof//3, 3), device=self.device)
        self.d3_visible = torch.zeros((self.num_envs), dtype=torch.int, device=self.device)
        self.im_ref_rb_target_pos = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device).float()
        self._build_mujoco_smpl_transform()

    def _build_mujoco_smpl_transform(self, ):
        mujoco_joint_names = [
            'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
            'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
            'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
            'R_Elbow', 'R_Wrist', 'R_Hand'
        ]
        self.smpl_2_mujoco = [
            joint_names.index(q) for q in mujoco_joint_names
            if q in joint_names
        ]
        self.mujoco_2_smpl = [
            mujoco_joint_names.index(q) for q in joint_names
            if q in mujoco_joint_names
        ]

    def build_body_tracking_mask(self, env_ids):
        ### build tracking_mask
        if self.has_tracking_mask:
            tracking_mask = torch.zeros((env_ids.shape[0], self.num_bodies), dtype=torch.int, device=self.device)
            selecte_num = torch.randint(self.num_bodies//2, self.num_bodies, (1,)).item()
            selected_idx = torch.randperm(self.num_bodies)[:selecte_num].to(self.device)
            tracking_mask[:, selected_idx] = 1
            tracking_mask[:, 0] = 1
            self.tracking_mask = tracking_mask.unsqueeze(-1)
        else:
            self.tracking_mask = torch.ones((env_ids.shape[0], self.num_bodies, 1), dtype=torch.int, device=self.device)

        # self.tracking_mask = torch.ones((env_ids.shape[0], self.num_bodies, 1), dtype=torch.int, device=self.device)
        if flags.test:
            self._upper_track_bodies_id = self._build_key_body_ids_tensor(self._upper_track_bodies)
            self.tracking_mask = torch.zeros((env_ids.shape[0], self.num_bodies, 1), device=self.device, dtype=torch.int)
            self.tracking_mask[:, self._upper_track_bodies_id] = 1
            self.tracking_mask[:, 0] = 1

    def _action_to_pd_targets(self, action):
        pd_tar = super()._action_to_pd_targets(action)
        return pd_tar

    def post_physics_step(self):
        super().post_physics_step()
        self._set_target_motion_state()

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        if not self.headless:
            self._build_infilling_marker(env_id, env_ptr)

    def _draw_task(self):
        self._update_marker()
        return
    
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
        if self.has_tracking_mask_obs:
            task_obs_detail.append(["tracking_mask", 24])
        return task_obs_detail

    def _update_marker(self):
        env_ids = torch.arange(self.num_envs).to(self.device)
        self._infilling_pos[:] = self.im_ref_rb_target_pos.clone()
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
        for i in range(self._num_joints):
            marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", self.num_envs + 5, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.8, 0.0))
            self._infilling_handles[env_id].append(marker_handle)

        return

    def _build_infilling_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._infilling_states = self._root_states.view(
            self.num_envs, num_actors,
            self._root_states.shape[-1])[..., 11:(11 + self._num_joints), :]
        self._infilling_pos = self._infilling_states[..., :3]
        self._infilling_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._infilling_handles, dtype=torch.int32, device=self.device)
        self._infilling_actor_ids = self._infilling_actor_ids.flatten()
        return


    def _load_motion(self, motion_file):
        assert (self._dof_offsets[-1] == self.num_dof)
        if self.smpl_humanoid:
            self._motion_lib = MotionLibSMPL(
                motion_file=motion_file,
                key_body_ids=self._key_body_ids.cpu().numpy(),
                device=self.device, masterfoot_conifg=self._masterfoot_config,
                min_length=self._min_motion_len, debug=flags.debug)
            self._motion_lib.load_motions(skeleton_trees = self.skeleton_trees, gender_betas = self.humanoid_betas.cpu(),
                limb_weights = self.humanoid_limb_and_weights.cpu(), random_sample=not flags.test)
            if self.use_different_motion_file:
                self._imitation_motion_lib = MotionLibSMPL(
                    motion_file=self.cfg['env']['imitation_motion_file'],
                    key_body_ids=self._key_body_ids.cpu().numpy(),
                    device=self.device, masterfoot_conifg=self._masterfoot_config,
                    min_length=self._min_motion_len, debug=flags.debug)
                self._imitation_motion_lib.load_motions(skeleton_trees = self.skeleton_trees, gender_betas = self.humanoid_betas.cpu(),
                    limb_weights = self.humanoid_limb_and_weights.cpu(), random_sample=not flags.test)
            else:
                self._imitation_motion_lib = self._motion_lib
            
        else:
            self._motion_lib = MotionLib(
                motion_file=motion_file,
                dof_body_ids=self._dof_body_ids,
                dof_offsets=self._dof_offsets,
                key_body_ids=self._key_body_ids.cpu().numpy(),
                device=self.device)
        self.motion_steps = (self._imitation_motion_lib._motion_lengths / self.dt).int()
        reference_motion_traj = self._imitation_motion_lib.gts[:, 0].clone()
        self.reference_start_index = np.minimum(np.random.randint(self.motion_steps.cpu().numpy()//16, self.motion_steps.cpu().numpy()//2, size = self.num_envs), self.max_episode_length//2)
        self.reference_start_index *= 0
        if self.cfg['args'].test:
            self.reference_start_index *= 0
        self.reference_length = np.minimum(np.random.randint(self.motion_steps.cpu().numpy()//2, self.motion_steps.cpu().numpy(), size = self.num_envs),  self.max_episode_length//2)
        self.motion_begin = torch.tensor(self.reference_start_index).to(self.device).int() + self._imitation_motion_lib.length_starts
        self.motion_end = self.motion_begin + torch.tensor(self.reference_length).to(self.device).int()
        self.reference_motion_traj = [reference_motion_traj[self.motion_begin[i]:self.motion_end[i]] for i in range(self.num_envs)]
        self.build_body_tracking_mask(torch.arange(self.num_envs, device=self.device, dtype=torch.int))
        return

    def resample_motions(self):
        print("Partial solution, only resample motions...")
        self._motion_lib.load_motions(skeleton_trees = self.skeleton_trees, limb_weights = self.humanoid_limb_and_weights.cpu(), gender_betas = self.humanoid_betas.cpu()) # For now, only need to sample motions since there are only 400 hmanoids
        if hasattr(self, '_imitation_motion_lib') and self.use_different_motion_file:
            self._imitation_motion_lib.load_motions(skeleton_trees = self.skeleton_trees, limb_weights = self.humanoid_limb_and_weights.cpu(), gender_betas = self.humanoid_betas.cpu())
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int)
        self.build_body_tracking_mask(env_ids)
    

    def _build_traj_generator(self):
        num_envs = self.num_envs
        episode_dur = self.max_episode_length * self.dt
        num_verts = 101
        dtheta_max = 2.0
        self._traj_gen = traj_generator.HybirdTrajGenerator(num_envs, episode_dur, num_verts,
                                                      self.device, dtheta_max,
                                                      self._speed_min, self._speed_max,
                                                      self._accel_max, self._sharp_turn_prob, self._traj_sample_timestep,
                                                      self.max_episode_length)

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        root_pos = self._humanoid_root_states[:, 0:3]
        self._traj_gen.reset(env_ids, root_pos, self.reference_motion_traj, self.reference_start_index)
        return None
    
    
    def _reset_task(self, env_ids):
        root_pos = self._humanoid_root_states[env_ids, 0:3]
        motion_steps = (self._imitation_motion_lib._motion_lengths / self.dt).int().cpu().numpy()
        reference_motion_traj = self._imitation_motion_lib.gts[:, 0].clone()
        reference_start_index = np.minimum(np.random.randint(motion_steps//16, motion_steps//2, size = self.num_envs), self.max_episode_length//2)
        reference_start_index *= 0
        if self.cfg['args'].test:
            reference_start_index *= 0
        reference_length = np.minimum(np.random.randint(motion_steps//2, motion_steps, size = self.num_envs),  self.max_episode_length//2)
        motion_begin = torch.tensor(reference_start_index).to(self.device).int() + self._imitation_motion_lib.length_starts
        motion_end = motion_begin + torch.tensor(reference_length).to(self.device).int()
        reference_motion_traj = [reference_motion_traj[motion_begin[i]:motion_end[i]] for i in range(self.num_envs)]
        ########### update the previous reference motion traj
        for i in env_ids:
            self.reference_start_index[i] = reference_start_index[i]
            self.reference_length[i] = reference_length[i]
            self.reference_motion_traj[i] = reference_motion_traj[i]
            self.motion_begin[i] = motion_begin[i]
            self.motion_end[i] = motion_end[i]
        self._traj_gen.reset(env_ids, root_pos, self.reference_motion_traj, self.reference_start_index)
        return


    def _sample_ref_state(self, env_ids, vel_min=1, vel_range=0.5):
        num_envs = env_ids.shape[0]
        motion_ids = env_ids
        motion_times = torch.tensor(self.reference_start_index).to(self.device)[env_ids] * self.dt

        if self.smpl_humanoid:
            curr_gender_betas = self.humanoid_betas[env_ids]
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot, body_vel, body_ang_vel = self._get_fixed_smpl_state_from_imitation_motionlib(
                motion_ids, motion_times, curr_gender_betas)
        else:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(
                motion_ids, motion_times)
            rb_pos, rb_rot = None, None


        if flags.random_heading:
            random_rot = np.zeros([num_envs, 3])
            random_rot[:, 2] = np.pi * (2 * np.random.random([num_envs]) - 1.0)
            random_heading_quat = torch.from_numpy(sRot.from_euler("xyz", random_rot).as_quat()).float().to(self.device)
            random_heading_quat_repeat = random_heading_quat[:, None].repeat(1, 24, 1)
            root_rot = quat_mul(random_heading_quat, root_rot).clone()
            rb_pos = quat_apply(random_heading_quat_repeat, rb_pos - root_pos[:, None, :]).clone()
            key_pos  = quat_apply(random_heading_quat_repeat[:, :4, :], (key_pos - root_pos[:, None, :])).clone()
            rb_rot = quat_mul(random_heading_quat_repeat, rb_rot).clone()
            root_ang_vel = quat_apply(random_heading_quat, root_ang_vel).clone()

            curr_heading = torch_utils.calc_heading_quat(root_rot)


            root_vel[:, 0] = torch.rand([num_envs]) * vel_range + vel_min
            root_vel = quat_apply(curr_heading, root_vel).clone()

        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot = self._sample_ref_state(env_ids)
        ## Randomrized location setting
        new_root_xy = self.terrain.sample_valid_locations(self.num_envs, env_ids)

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
        root_pos[:, 2] += 0.03
        key_pos[..., 0:2] += diff_xy[:, None, :]
        key_pos[...,  2] += center_height[:, None]

        rb_pos[..., 0:2] += diff_xy[:, None, :]
        rb_pos[..., 2] += center_height[:, None]
        rb_pos[..., 2] += 0.03
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
    
        use_env_ids = not (len(env_ids) == self.num_envs and torch.all(env_ids == torch.arange(self.num_envs, device=self.device)))
        if not use_env_ids:
            self._set_target_motion_state()
        else:
            self._set_target_motion_state(env_ids)
        return
    
    def _get_fixed_smpl_state_from_imitation_motionlib(self, motion_ids, motion_times, curr_gender_betas):
        # Used for intialization. Not used for sampling. Only used for AMP, not imitation.
        motion_res = self._get_smpl_state_from_imitation_motionlib_cache(motion_ids, motion_times)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, _, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        with torch.no_grad():
            gender = curr_gender_betas[:, 0]
            betas = curr_gender_betas[:, 1:]
            B, _ = betas.shape

            genders_curr = gender == 2
            height_tolorance = 0.02
            if genders_curr.sum() > 0:
                poses_curr = pose_aa[genders_curr]
                root_pos_curr = root_pos[genders_curr]
                betas_curr = betas[genders_curr]
                vertices_curr, joints_curr = self.smpl_parser_f.get_joints_verts(
                    poses_curr, betas_curr, root_pos_curr)
                offset = joints_curr[:, 0] - root_pos[genders_curr]
                diff_fix = ((vertices_curr - offset[:, None])[..., -1].min(dim=-1).values - height_tolorance)
                root_pos[genders_curr, ..., -1] -= diff_fix
                key_pos[genders_curr, ..., -1] -= diff_fix[:, None]
                rb_pos[genders_curr, ..., -1] -= diff_fix[:, None]

            genders_curr = gender == 1
            if genders_curr.sum() > 0:
                poses_curr = pose_aa[genders_curr]
                root_pos_curr = root_pos[genders_curr]
                betas_curr = betas[genders_curr]
                vertices_curr, joints_curr = self.smpl_parser_m.get_joints_verts(
                    poses_curr, betas_curr, root_pos_curr)

                offset = joints_curr[:, 0] - root_pos[genders_curr]
                diff_fix = (
                    (vertices_curr - offset[:, None])[..., -1].min(dim=-1).values -
                    height_tolorance)
                root_pos[genders_curr, ..., -1] -= diff_fix
                key_pos[genders_curr, ..., -1] -= diff_fix[:, None]
                rb_pos[genders_curr, ..., -1] -= diff_fix[:, None]

            genders_curr = gender == 0
            if genders_curr.sum() > 0:
                poses_curr = pose_aa[genders_curr]
                root_pos_curr = root_pos[genders_curr]
                betas_curr = betas[genders_curr]
                vertices_curr, joints_curr = self.smpl_parser_n.get_joints_verts(
                    poses_curr, betas_curr, root_pos_curr)

                offset = joints_curr[:, 0] - root_pos[genders_curr]
                diff_fix = (
                    (vertices_curr - offset[:, None])[..., -1].min(dim=-1).values -
                    height_tolorance)
                root_pos[genders_curr, ..., -1] -= diff_fix
                key_pos[genders_curr, ..., -1] -= diff_fix[:, None]
                rb_pos[genders_curr, ..., -1] -= diff_fix[:, None]

            return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot, body_vel, body_ang_vel

    def _get_smpl_state_from_imitation_motionlib_cache(self, motion_ids, motion_times):
        ## Chace the motion
        motion_res = self._imitation_motion_lib.get_motion_state_smpl(motion_ids, motion_times)
        return motion_res
    
    def _set_target_motion_state(self, env_ids=None):
        if env_ids is None:
            use_env_ids = False
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            use_env_ids = True
        ref_start = torch.tensor(self.reference_start_index).to(self.device)
        ref_length = torch.tensor(self.reference_length).to(self.device)
        motion_times = (self.progress_buf[env_ids] + 1) * self.dt

        motion_res = self._get_smpl_state_from_imitation_motionlib_cache(env_ids, motion_times)
        time_steps = 1
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"]
        
        if flags.test:
            d3_visible = torch.zeros((env_ids.shape[0]), dtype=torch.int, device=self.device)
            mask = ((self.progress_buf[env_ids] + 1) >= ref_start[env_ids]) & ((self.progress_buf[env_ids] + 1) < ref_start[env_ids] + ref_length[env_ids])
            d3_visible[mask] = 1

            if not use_env_ids:
                self.d3_visible = d3_visible
                self._target_dof_pos = dof_pos
            else:
                self.d3_visible[env_ids] = d3_visible
                self._target_dof_pos[env_ids] = dof_pos


    def _compute_task_obs(self, env_ids=None):
        # Compute task observations (terrain, trajectory, self state)
        basic_obs = super()._compute_task_obs(env_ids)
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

        ref_start = torch.tensor(self.reference_start_index).to(self.device)
        ref_length = torch.tensor(self.reference_length).to(self.device)
        motion_times = (self.progress_buf[env_ids] + 1) * self.dt
        curr_gender_betas = self.humanoid_betas[env_ids]

        motion_res = self._get_smpl_state_from_imitation_motionlib_cache(env_ids, motion_times)
        time_steps = 1
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"]

        d3_visible = torch.zeros((env_ids.shape[0]), dtype=torch.int, device=self.device)
        mask = ((self.progress_buf[env_ids] + 1) >= ref_start[env_ids]) & ((self.progress_buf[env_ids] + 1) < ref_start[env_ids] + ref_length[env_ids])
        d3_visible[mask] = 1

        ref_root_pos = root_pos.clone()
        ref_root_rot = root_rot.clone()
        ref_rb_pos = ref_rb_pos.clone()
        ref_rb_rot = ref_rb_rot.clone()
        ref_body_vel = ref_body_vel.clone()
        ref_diff_xy = self._traj_gen.get_diff_xy(env_ids)
        ref_root_pos[:, 0:2] += ref_diff_xy
        ref_rb_pos[:, :, 0:2] += ref_diff_xy.unsqueeze(1)
        root_state = torch.cat([ref_root_pos, ref_root_rot], dim=-1)
        center_heights = self.get_center_heights(root_states=root_state, env_ids=env_ids)
        center_heights = center_heights.mean(dim=-1, keepdim=True)
        ref_rb_pos[..., 2] += center_heights

        ########### update for visualization
        if flags.test:  
            self.im_ref_rb_target_pos[env_ids] = ref_rb_pos.clone()
        
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

        self.im_dof_pos[env_ids] = dof_pos.clone()

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
        basic_obs = super()._compute_flip_task_obs(normal_task_obs, env_ids)
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

        ref_start = torch.tensor(self.reference_start_index).to(self.device)
        ref_length = torch.tensor(self.reference_length).to(self.device)
        motion_times = (self.progress_buf[env_ids] + 1) * self.dt
        motion_res = self._get_smpl_state_from_imitation_motionlib_cache(env_ids, motion_times)
        time_steps = 1
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"]

        d3_visible = torch.zeros((env_ids.shape[0]), dtype=torch.int, device=self.device)
        mask = ((self.progress_buf[env_ids] + 1) >= ref_start[env_ids]) & ((self.progress_buf[env_ids] + 1) < ref_start[env_ids] + ref_length[env_ids])
        d3_visible[mask] = 1

        ################## prepare for imitation
        root_pos = root_pos.clone()
        ref_rb_pos = ref_rb_pos.clone()
        ref_rb_rot = ref_rb_rot.clone()
        ref_body_vel = ref_body_vel.clone()
        root_states = torch.cat([root_pos, root_rot], dim=-1).clone()
        ref_diff_xy = self._traj_gen.get_diff_xy(env_ids)
        root_pos[:, 0:2] += ref_diff_xy
        ref_rb_pos[:, :, 0:2] += ref_diff_xy.unsqueeze(1)
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


    def _compute_im_rewards(self, actions):

        env_ids = torch.arange(self.num_envs,
                               dtype=torch.long,
                               device=self.device)
        body_pos = self._rigid_body_pos.clone()
        body_rot = self._rigid_body_rot.clone()
        body_dof_pos = self._dof_pos.clone()
        body_dof_vel = self._dof_vel.clone()

        ref_start = torch.tensor(self.reference_start_index).to(self.device)
        ref_length = torch.tensor(self.reference_length).to(self.device)
        motion_times = (self.progress_buf[env_ids]) * self.dt
        motion_res = self._get_smpl_state_from_imitation_motionlib_cache(env_ids, motion_times)
  
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"]

        ### prepare for reward
        root_pos = root_pos.clone()
        ref_rb_pos = ref_rb_pos.clone()
        ref_diff_xy = self._traj_gen.get_diff_xy(env_ids)
        root_pos[:, 0:2] += ref_diff_xy
        ref_rb_pos[:, :, 0:2] += ref_diff_xy.unsqueeze(1)
        root_state = torch.cat([root_pos, root_rot], dim=-1)
        center_heights = self.get_center_heights(root_states=root_state, env_ids=env_ids)
        center_heights = center_heights.mean(dim=-1, keepdim=True)
        ref_rb_pos[..., 2] += center_heights

        d3_visible = torch.zeros((self.num_envs), dtype=torch.int, device=self.device)
        mask = ((self.progress_buf[env_ids]) >= ref_start[env_ids]) & ((self.progress_buf[env_ids]) < ref_start[env_ids] + ref_length[env_ids])
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

        ref_start = torch.tensor(self.reference_start_index).to(self.device)
        ref_length = torch.tensor(self.reference_length).to(self.device)
        motion_times = (self.progress_buf[env_ids]) * self.dt
        motion_res = self._get_smpl_state_from_imitation_motionlib_cache(env_ids, motion_times)
  
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"]

        root_pos = root_pos.clone()
        ref_rb_pos = ref_rb_pos.clone()
        ref_diff_xy = self._traj_gen.get_diff_xy(env_ids)
        root_pos[:, 0:2] += ref_diff_xy
        ref_rb_pos[:, :, 0:2] += ref_diff_xy.unsqueeze(1)
        center_heights = self.get_center_heights(root_states=root_states, env_ids=env_ids)
        center_heights = center_heights.mean(dim=-1, keepdim=True)
        ref_rb_pos[..., 2] += center_heights

        d3_visible = torch.zeros((self.num_envs), dtype=torch.int, device=self.device)
        mask = ((self.progress_buf[env_ids]) >= ref_start[env_ids] + self.reset_buffer) & ((self.progress_buf[env_ids]) < ref_start[env_ids] + ref_length[env_ids])
        d3_visible[mask] = 1

        tracking_mask = self.tracking_mask[:, self._reset_body_ids]
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf, self.progress_buf, self._contact_forces,
            self._contact_body_ids, center_height, self._rigid_body_pos[:, self._reset_body_ids] ,
              ref_rb_pos[:, self._reset_body_ids], d3_visible, tracking_mask,
            tar_pos, self.max_episode_length, self._fail_dist,
            self._enable_early_termination, self._termination_heights, flags.no_collision_check,
            self.terminate_dist[0], self.use_imitation_reset)
        return
    
    def get_current_pose(self):
        # hack for rendering
        ############ begin to prepare physics states ###########
        root_pos = self._rigid_body_pos[:, 0, :3]
        root_rot = self._rigid_body_rot[:, 0, :4]
        physics_kp = self._rigid_body_pos[:, self.mujoco_2_smpl, :3]
        root_rot = quaternion_to_angle_axis(root_rot[..., [3, 0, 1, 2]])
        dof_pos = self._dof_pos
        B = dof_pos.shape[0]
        dof_pos = dof_pos.reshape(-1, 3)
        angle, axis = torch_utils.exp_map_to_angle_axis(dof_pos)
        dof_pos = (axis * angle.unsqueeze(-1)).reshape(B, -1, 3)
        all_rot = torch.cat([root_rot.unsqueeze(1), dof_pos], dim=1)[:, self.mujoco_2_smpl]
        root_rot = all_rot[:, 0, :]
        dof_pos = all_rot[:, 1:, :].reshape(-1, 23, 3)
        ########### end to prepare physics states ###########

        ########### begin to prepare reference states ###########
        body_shapes =  torch.from_numpy(self._amass_gender_betas[:, :])

        trajectory_target = self.traj_samples
        imitation_target = self.im_ref_rb_target_pos[:, self.mujoco_2_smpl, :3]
        error = (physics_kp - physics_kp[:, :1]) - (imitation_target - imitation_target[:, :1])
        error = ((error**2).sum(dim=2))**0.5
        error = error.mean()
        d3_visible = self.d3_visible
        tracking_mask = self.tracking_mask[:, self.mujoco_2_smpl]
        

        ########### end to prepare reference states ###########
        return root_pos, root_rot, dof_pos, physics_kp, trajectory_target, imitation_target, d3_visible, tracking_mask, body_shapes

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



@torch.jit.script
def compute_keypoint_imitation_observations(root_pos, root_rot, body_pos, ref_body_pos, betas, d3_observation, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, bool) -> Tensor
    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    ############ due to the real to sim assignment, we only use the xy obs for position

    ############# root relative state #############
    ############# diff root height #############
    target_root_pos = ref_body_pos[:, 0, :]

    ############# rel root ref obs #############
    heading_rot, heading = torch_utils.calc_heading_quat_inv_with_heading(root_rot)
    ############# rel 2d pos #############
    target_rel_pos = target_root_pos[:, :3] - root_pos[:, :3]
    target_rel_pos = torch_utils.my_quat_rotate(heading_rot, target_rel_pos)
    target_rel_2d_pos = target_rel_pos[:, :2]

    ############# diff body pos #############
    target_rel_root_body_pos = ref_body_pos - target_root_pos.view(B, 1, 3)
    rel_root_body_pos = body_pos - root_pos.view(B, 1, 3)
    target_rel_pos = target_rel_root_body_pos - rel_root_body_pos
    num_joints = target_rel_pos.shape[1]
    target_rel_pos = target_rel_pos.reshape(-1, 3)
    target_rel_pos = torch_utils.my_quat_rotate(heading_rot[:, None, :].repeat(1, J, 1).reshape(-1, 4), target_rel_pos)
    target_rel_pos = target_rel_pos.reshape(B, -1, 3)
    target_rel_pos = target_rel_pos[:, 1:]

    ############# whole body relative state #############
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat( (1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_global_body_pos[..., 2] *= 0
    diff_local_body_pos = torch_utils.my_quat_rotate(heading_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3)).reshape(B, time_steps, J, 3)

    if (d3_observation==0).sum() >0:
        diff_local_body_pos[d3_observation==0] *= 0
        target_rel_2d_pos[d3_observation==0] *= 0
        target_rel_pos[d3_observation==0] *= 0


    diff_local_body_pos_flat = diff_local_body_pos.view(-1, 3)[:, :2].contiguous()
    obs.append(target_rel_2d_pos.view(B, -1))
    obs.append(target_rel_pos.view(B, -1))
    obs.append(diff_local_body_pos_flat.view(B, -1)) # 1 * 10 * 3 * 3
    obs.append(betas.view(B, -1))
    obs = torch.cat(obs, dim=-1)
    return obs


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