# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from uuid import uuid4
import numpy as np
import os

import torch
import multiprocessing

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import joblib
from utils import torch_utils
from uhc.smpllib.smpl_local_robot import Robot

from amp.utils.flags import flags
from env.tasks.base_task import BaseTask
from tqdm import tqdm
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from collections import defaultdict
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from amp.utils.draw_utils import agt_color

ENABLE_MAX_COORD_OBS = True
# PERTURB_OBJS = [
#     ["small", 60],
#     ["small", 7],
#     ["small", 10],
#     ["small", 35],
#     ["small", 2],
#     ["small", 2],
#     ["small", 3],
#     ["small", 2],
#     ["small", 2],
#     ["small", 3],
#     ["small", 2],
#     ["large", 60],
#     ["small", 300],
# ]
PERTURB_OBJS = [
    ["small", 60],
]


class Humanoid(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id,
                 headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.has_task = False


        self.load_smpl_configs(cfg)

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._enable_early_termination = self.cfg["env"][
            "enableEarlyTermination"]

        self.key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(self.key_bodies)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        self.dt = self.control_freq_inv * sim_params.dt
        self._setup_tensors()
        self.reward_raw = torch.zeros((self.num_envs, 1)).to(self.device)

        return

    def _load_proj_asset(self):
        asset_root = "amp/data/assets/mjcf/"

        # small_asset_file = "block_projectile.urdf"
        small_asset_file = "ball_medium.urdf"
        small_asset_options = gymapi.AssetOptions()
        small_asset_options.angular_damping = 0.01
        small_asset_options.linear_damping = 0.01
        small_asset_options.max_angular_velocity = 100.0
        small_asset_options.density = 200.0
        small_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._small_proj_asset = self.gym.load_asset(self.sim, asset_root, small_asset_file, small_asset_options)

        large_asset_file = "block_projectile_large.urdf"
        large_asset_options = gymapi.AssetOptions()
        large_asset_options.angular_damping = 0.01
        large_asset_options.linear_damping = 0.01
        large_asset_options.max_angular_velocity = 100.0
        large_asset_options.density = 100.0
        large_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._large_proj_asset = self.gym.load_asset(self.sim, asset_root, large_asset_file, large_asset_options)
        return

    def _build_proj(self, env_id, env_ptr):
        for i, obj in enumerate(PERTURB_OBJS):
            default_pose = gymapi.Transform()
            default_pose.p.x = 260
            default_pose.p.y = 50
            default_pose.p.z = -5
            obj_type = obj[0]
            if (obj_type == "small"):
                proj_asset = self._small_proj_asset
            elif (obj_type == "large"):
                proj_asset = self._large_proj_asset

            proj_handle = self.gym.create_actor(env_ptr, proj_asset, default_pose, "proj{:d}".format(i), env_id, 2)
            self._proj_handles.append(proj_handle)

        return

    def _setup_tensors(self):
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(
            self.sim)

        # ZL: needs to put this back
        # sensors_per_env = 2
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
            self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()

        self._humanoid_root_states = self._root_states.view(
            self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0

        self._humanoid_actor_ids = num_actors * torch.arange(
            self.num_envs, device=self.device, dtype=torch.int32)

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env,
                                             2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env,
                                             2)[..., :self.num_dof, 1]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos,
                                                 device=self.device,
                                                 dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel,
                                                 device=self.device,
                                                 dtype=torch.float)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self._rigid_body_pos = self._rigid_body_state_reshaped[..., :self.num_bodies,
                                                         0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., :self.num_bodies,
                                                         3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., :self.num_bodies,
                                                         7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[
            ..., :self.num_bodies, 10:13]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]

        self._terminate_buf = torch.ones(self.num_envs,
                                         device=self.device,
                                         dtype=torch.long)

        self._build_termination_heights()
        contact_bodies = self.cfg["env"]["contactBodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(self.key_bodies)

        self._contact_body_ids = self._build_contact_body_ids_tensor(
            contact_bodies)

        if self.viewer != None:
            self._init_camera()

        ###################################
        if self.has_flip_observation:
            self._flip_obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)

    def load_smpl_configs(self, cfg):
        self.smpl_humanoid = (cfg["env"]["asset"]['assetFileName'] == "mjcf/smpl_humanoid.xml") or (cfg["env"]["asset"]["assetFileName"] == "mjcf/smpl_mesh_humanoid_amass_v1.xml")
        self._bias_offset = cfg["env"].get("bias_offset", False)
        self._divide_group = cfg["env"].get("divide_group", False)
        self._group_obs = cfg["env"].get("group_obs", False)
        self._disable_group_obs = cfg["env"].get("disable_group_obs", False)
        if self._divide_group:
            self._group_num_people = group_num_people = min(cfg['env'].get("num_env_group", 128), cfg['env']['numEnvs'])
            self._group_ids = torch.tensor(np.arange(cfg["env"]["numEnvs"] / group_num_people).repeat(group_num_people).astype(int))

        self._has_shape_obs = cfg["env"].get("has_shape_obs", False)
        self._has_shape_obs_disc = cfg["env"].get("has_shape_obs_disc", False)
        self._has_limb_weight_obs = cfg["env"].get("has_weight_obs", False)
        self._has_limb_weight_obs_disc = cfg["env"].get("has_weight_obs_disc", False)

        self.has_shape_variation = cfg["env"].get("has_shape_variation", False)
        self._has_self_collision = cfg["env"].get("has_self_collision", False)
        self._has_mesh = cfg["env"].get("has_mesh", True)
        self._replace_feet = cfg["env"].get("replace_feet", True) # replace feet or not
        self._has_jt_limit = cfg["env"].get("has_jt_limit", True)
        self._has_dof_subset = cfg["env"].get("has_dof_subset", False)
        self._has_upright_start = cfg["env"].get("has_upright_start", True)
        self._has_smpl_pd_offset = cfg["env"].get("has_smpl_pd_offset", False)
        self.remove_toe = cfg["env"].get("remove_toe", False)
        self.big_ankle = cfg["env"].get("big_ankle", False)
        self._remove_thorax = cfg["env"].get("remove_thorax", False)
        self._real_weight_porpotion = cfg["env"].get("real_weight_porpotion", False)
        self._real_weight = cfg["env"].get("real_weight", False)
        self.shape_resampling_interval = cfg["env"].get(
            "shape_resampling_interval", 100)
        self._remove_ankles = cfg["env"].get("remove_ankles", False)
        self._remove_neck = cfg["env"].get("remove_neck", False)
        self.getup_schedule = cfg["env"].get("getup_schedule", False)
        self._kp_scale = cfg["env"].get("kp_scale", 1.0)
        self._kd_scale = cfg["env"].get("kd_scale", self._kp_scale)
        self._masterfoot  = cfg["env"].get("masterfoot", False)
        self._master_range = cfg["env"].get("master_range", 30)
        self._freeze_toe = cfg["env"].get("freeze_toe", True)
        self.motion_sym_loss = cfg["env"].get("motion_sym_loss", False)
        self.has_flip_observation = cfg['env'].get("hasFlipObservation", False)
        if self.motion_sym_loss:
            self.has_flip_observation = True
        self.remove_toe_im = cfg["env"].get("remove_toe_im", True)
        self.remove_foot_reset_im = cfg["env"].get("remove_foot_reset_im", True)
        self.velocity_map = cfg["env"].get("velocity_map", False)
        self.hard_negative = cfg["env"].get("hard_negative", False) # hard negative sampling for im
        self.cycle_motion = cfg["env"].get("cycle_motion",
                                           False)  # Cycle motion to reach 300

        disc_idxes = []
        self._body_names_orig = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
        _body_names_orig_copy = self._body_names_orig.copy()
        if self.remove_toe_im:
            _body_names_orig_copy.remove('L_Toe'); _body_names_orig_copy.remove('R_Toe')
        self._full_track_bodies = _body_names_orig_copy
        self._upper_track_bodies = ['Pelvis','Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']


        if self._masterfoot:
            self._body_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'L_Toe_1', 'L_Toe_1_1', 'L_Toe_2', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'R_Toe_1', 'R_Toe_1_1', 'R_Toe_2', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
            self._body_to_orig = [self._body_names.index(name) for name in self._body_names_orig]
            self._body_to_orig_without_toe = [self._body_names.index(name) for name in self._body_names_orig if name not in ['L_Toe', 'R_Toe']]
            self.orig_to_orig_without_toe = [ self._body_names_orig.index(name) for name in self._body_names_orig if name not in ['L_Toe', 'R_Toe']]

            self._masterfoot_config = {
                "body_names_orig": self._body_names_orig,
                "body_names": self._body_names,
                "body_to_orig": self._body_to_orig,
                "body_to_orig_without_toe": self._body_to_orig_without_toe,
                "orig_to_orig_without_toe": self.orig_to_orig_without_toe,
            }
        else:
            self._body_names = self._body_names_orig
            self._masterfoot_config = None



        self._dof_names = self._body_names[1:]
        remove_names = ["L_Hand", "R_Hand", "L_Toe", "R_Toe"]
        if self._remove_ankles:
            remove_names.append("L_Ankle")
            remove_names.append("R_Ankle")
        if self._remove_thorax:
            remove_names.append("L_Thorax")
            remove_names.append("R_Thorax")
        if self._remove_neck:
            remove_names.append("Neck")
            remove_names.append("Head")

        if self._masterfoot:
            remove_names += ["L_Toe_1", "L_Toe_1_1", "L_Toe_2", "R_Toe_1", "R_Toe_1_1", "R_Toe_2"]

        if self._masterfoot:
            self.limb_weight_group = [['L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'L_Toe_1', 'L_Toe_1_1', 'L_Toe_2'], \
                                        ['R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'R_Toe_1', 'R_Toe_1_1', 'R_Toe_2'], \
                                            ['L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand'], \
                                            ['R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'], \
                                                ['Pelvis',   'Torso', 'Spine', 'Chest', 'Neck', 'Head',  ]]
            self.limb_weight_group = [[self._body_names.index(g) for g in group] for group in self.limb_weight_group]
        else:
            self.limb_weight_group = [
                ['L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe'], \
                    ['R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe'], \
                        ['Pelvis',  'Torso', 'Spine', 'Chest', 'Neck', 'Head'], \
                            [ 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand'], \
                                ['R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']]
            self.limb_weight_group = [[self._body_names.index(g) for g in group] for group in self.limb_weight_group]



        for idx, name in enumerate(self._dof_names):
            if not name in remove_names:
                disc_idxes.append(np.arange(idx * 3, (idx + 1) * 3))

        self.dof_subset = torch.from_numpy(np.concatenate(disc_idxes))
        self.selected_group_jts_names = ['Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Ankle', 'R_Ankle', 'L_Elbow', 'R_Elbow', 'L_Hand', 'R_Hand']
        self.selected_group_jts = torch.tensor([self._body_names.index(jt_name) for jt_name in self.selected_group_jts_names])

        if self._masterfoot:
            self.left_to_right_index = [0, 8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 20, 21, 22, 23, 24]
            self.left_to_right_index_action = [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 24, 25, 26, 27, 28, 19, 20, 21, 22, 23]
        else:
            self.left_to_right_index = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 14, 15, 16, 17, 18]
            self.left_to_right_index_action = [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11, 12, 18, 19, 20, 21, 22, 13, 14, 15, 16, 17]

        if self.smpl_humanoid:
            self._load_amass_gender_betas()

    def _clear_recorded_states(self):
        del self.state_record
        self.state_record = defaultdict(list)

    def _record_states(self):
        self.state_record['dof_pos'].append(self._dof_pos.cpu().clone())
        self.state_record['root_states'].append(self._humanoid_root_states.cpu().clone())
        self.state_record['progress'].append(self.progress_buf.cpu().clone())

    def _write_states_to_file(self, file_name):

        self.state_record['skeleton_trees'] = self.skeleton_trees
        self.state_record['humanoid_betas'] = self.humanoid_betas
        if self.num_envs > 32:
            print("too many enviroment!!! dumping will be too slow....")
            return
        else:
            print(f"Dumping states into {file_name}")

        progress = torch.stack(self.state_record['progress'], dim = 1)
        progress_diff = torch.cat([progress, -10 * torch.ones(progress.shape[0], 1).to(progress)], dim = -1)

        diff = torch.abs(progress_diff[:, :-1] - progress_diff[:, 1:])
        split_idx = torch.nonzero(diff > 1)
        split_idx[:, 1] += 1
        dof_pos_all =  torch.stack(self.state_record['dof_pos'])
        root_states_all =  torch.stack(self.state_record['root_states'])
        fps = 60
        motion_dict_dump = {}
        num_for_this_humanoid = 0
        curr_humanoid_index = 0

        for idx in tqdm(range(len(split_idx))):
            split_info = split_idx[idx]
            humanoid_index = split_info[0]

            if humanoid_index != curr_humanoid_index:
                num_for_this_humanoid = 0
                curr_humanoid_index = humanoid_index

            if num_for_this_humanoid == 0:
                start = 0
            else:
                start = split_idx[idx - 1][-1]

            end = split_idx[idx][-1]

            dof_pos_seg = dof_pos_all[start: end, humanoid_index]
            B, H= dof_pos_seg.shape

            root_states_seg = root_states_all[start: end, humanoid_index]
            body_quat = torch.cat([root_states_seg[:, None, 3:7], torch_utils.exp_map_to_quat(dof_pos_seg.reshape(B, -1, 3))], dim=1)

            # skeleton_state = SkeletonState.from_rotation_and_root_translation(
            #     self.state_record['skeleton_trees'][humanoid_index],
            #     body_quat,
            #     root_states_seg[:, :3],
            #     is_local=True)
            # skeleton_motion = SkeletonMotion.from_skeleton_state(skeleton_state, fps=fps)
            # motion_dump = skeleton_motion.to_dict()

            motion_dump = {
                "skeleton_tree": self.state_record['skeleton_trees'][humanoid_index].to_dict(),
                "body_quat": body_quat,
                "trans": root_states_seg[:, :3],
            }
            motion_dump['fps'] = fps
            motion_dump['betas'] = self.humanoid_betas[humanoid_index].detach().cpu().numpy()
            motion_dict_dump[f"{humanoid_index}_{num_for_this_humanoid}"] = motion_dump
            num_for_this_humanoid += 1

        joblib.dump(motion_dict_dump, file_name)
        self.state_record = defaultdict(list)


    def get_obs_size(self):
        return self._num_self_obs

    def get_self_obs_size(self):
        return self._num_self_obs

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')

        self.sim = super().create_sim(self.device_id, self.graphics_device_id,
                                      self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'],
                          int(np.sqrt(self.num_envs)))
        return

    def reset(self, env_ids=None):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs),
                               device=self.device,
                               dtype=torch.long)
        self._reset_envs(env_ids)
        return

    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL,
                    gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
        return

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self._contact_forces[env_ids] = 0
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction

        # plane_params.static_friction = 50
        # plane_params.dynamic_friction = 50

        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _setup_character_props(self, key_bodies):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            ### ZL: changes
            self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
            self._dof_offsets = [
                0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28
            ]
            self._dof_obs_size = 72
            self._num_actions = 28

            if (ENABLE_MAX_COORD_OBS):
                self._num_self_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
            else:
                self._num_self_obs = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

        elif (asset_file == "mjcf/smpl_humanoid.xml"):
            # import ipdb; ipdb.set_trace()
            self._dof_body_ids = np.arange(1, len(self._body_names))
            self._dof_offsets = np.linspace(0,
                                            len(self._dof_names) * 3,
                                            len(self._body_names)).astype(int)
            self._dof_obs_size = len(self._dof_names) * 6
            self._num_actions = len(self._dof_names) * 3
            if (ENABLE_MAX_COORD_OBS):
                self._num_self_obs = 1 + len(self._body_names) * (3 + 6 + 3 + 3) - 3 # height + num_bodies * 15 (pos + vel + rot + ang_vel) - root_pos
            else:
                raise NotImplementedError()

            if self._has_shape_obs: self._num_self_obs += 11 - 6
            # if self._has_limb_weight_obs: self._num_self_obs += 23 + 24 if not self._masterfoot else  29 + 30 # 23 + 24 (length + weight)
            if self._has_limb_weight_obs: self._num_self_obs += 10

            if not self._root_height_obs: self._num_self_obs -= 1
        
        elif (asset_file == "mjcf/smpl_mesh_humanoid_amass_v1.xml"):
            # import ipdb; ipdb.set_trace()
            self._dof_body_ids = np.arange(1, len(self._body_names))
            self._dof_offsets = np.linspace(0,
                                            len(self._dof_names) * 3,
                                            len(self._body_names)).astype(int)
            self._dof_obs_size = len(self._dof_names) * 6
            self._num_actions = len(self._dof_names) * 3
            if (ENABLE_MAX_COORD_OBS):
                self._num_self_obs = 1 + len(self._body_names) * (3 + 6 + 3 + 3) - 3 # height + num_bodies * 15 (pos + vel + rot + ang_vel) - root_pos
            else:
                raise NotImplementedError()

            if self._has_shape_obs: self._num_self_obs += 11 - 6
            # if self._has_limb_weight_obs: self._num_self_obs += 23 + 24 if not self._masterfoot else  29 + 30 # 23 + 24 (length + weight)
            if self._has_limb_weight_obs: self._num_self_obs += 10

            if not self._root_height_obs: self._num_self_obs -= 1

        elif (asset_file == "mjcf/amp_humanoid_sword_shield.xml"):
            self._dof_body_ids = [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16]
            self._dof_offsets = [
                0, 3, 6, 9, 10, 13, 16, 17, 20, 21, 24, 27, 28, 31
            ]
            self._dof_obs_size = 78
            self._num_actions = 31

            if (ENABLE_MAX_COORD_OBS):
                self._num_self_obs = 1 + 17 * (3 + 6 + 3 + 3) - 3
            else:
                self._num_self_obs = 13 + self._dof_obs_size + 31 + 3 * num_key_bodies

        elif (asset_file == "mjcf/ov_humanoid.xml"):
            self._dof_body_ids = [
                1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17
            ]
            self._dof_offsets = [
                0, 3, 6, 9, 12, 15, 16, 19, 22, 23, 26, 27, 30, 33, 34, 37
            ]
            self._dof_obs_size = 90
            self._num_self_obs = 13 + self._dof_obs_size + 37 + 3 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
            self._num_actions = 37
        elif (asset_file == "mjcf/ov_humanoid_sword_shield.xml"):
            self._dof_body_ids = [
                1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 14, 15, 16, 17, 18, 19
            ]
            self._dof_offsets = [
                0, 3, 6, 9, 12, 15, 16, 19, 22, 25, 26, 29, 30, 33, 36, 37, 40
            ]
            self._dof_obs_size = 96
            self._num_self_obs = 13 + self._dof_obs_size + 40 + 3 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
            self._num_actions = 40
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert (False)

        return

    def _build_termination_heights(self):
        head_term_height = 0.3
        shield_term_height = 0.32

        termination_height = self.cfg["env"]["terminationHeight"]
        self._termination_heights = np.array([termination_height] *
                                             self.num_bodies)

        head_id = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles[0], "head")
        self._termination_heights[head_id] = max(
            head_term_height, self._termination_heights[head_id])

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        if (asset_file == "mjcf/amp_humanoid_sword_shield.xml"):
            left_arm_id = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.humanoid_handles[0], "left_lower_arm")
            self._termination_heights[left_arm_id] = max(
                shield_term_height, self._termination_heights[left_arm_id])

        self._termination_heights = to_torch(self._termination_heights,
                                             device=self.device)
        return

    def _create_smpl_humanoid_xml(self, num_humanoids, smpl_robot, queue, pid):
        np.random.seed(np.random.randint(5002) * (pid + 1))
        res = {}
        for idx in tqdm(num_humanoids):
            asset_id = uuid4()
            asset_file_real = f"/tmp/smpl/smpl_humanoid_{asset_id}.xml"
            asset_file_real_orig = f"/tmp/smpl/smpl_humanoid_{asset_id}_orig.xml"

            if self.has_shape_variation:
                #####
                gender_beta = self._amass_gender_betas[idx % self._amass_gender_betas.shape[0]]
                # gender_beta = self._amass_gender_betas[(idx + 10) % self._amass_gender_betas.shape[0]]
                #####
                # gender_shape_id = np.random.randint(0, self._amass_gender_betas.shape[0])
                # gender_beta = self._amass_gender_betas[idx % self._amass_gender_betas.shape[0]]
                # gender_beta = self._amass_gender_betas[0]; print('debug: using same beta'); print('debug: using same beta'); print('debug: using same beta'); print('debug: using same beta'); print('debug: using same beta'); print('debug: using same beta'); print('debug: using same beta')
                ######

                # if flags.server_mode :
                # list_to_test = [0, 5, 7, 9]
                # gender_beta = self._amass_gender_betas[list_to_test[idx % len(list_to_test)]]
                # gender_beta[1:2] = gender_beta[1:2] * 2.5
                # gender_beta = self._amass_gender_betas[-1]
            else:
                gender_beta = np.zeros(17)
            smpl_robot.load_from_skeleton(betas=torch.from_numpy(gender_beta[None, 1:]),
                                          gender=gender_beta[0:1],
                                          objs_info=None)
            smpl_robot.write_xml(asset_file_real)

            # asset_file_real = "test.xml"
            res[idx] = (gender_beta, asset_file_real)

        if not queue is None:
            queue.put(res)
        else:
            return res

    def _load_amass_gender_betas(self):
        if self.cfg['args'].test:
            self._amass_gender_betas = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        else:
            if self._has_mesh:
                gender_betas_data = joblib.load("data/amass/pkls/amass_isaac_gender_betas_unique.pkl")
                self._amass_gender_betas = np.array(gender_betas_data)[:, :11]
            else:
                gender_betas_data = joblib.load("data/amass/pkls/amass_isaac_gender_betas_unique.pkl")
                #gender_betas_data = joblib.load("data/embodied_poses/pkls/betas.pkl")
                self._amass_gender_betas = np.array(gender_betas_data)[:, :11]

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        # asset_root = "./"
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        self.humanoid_masses = []
        self.humanoid_files = dict()

        if (self.smpl_humanoid):
            self.humanoid_betas = []
            self.humanoid_assets = []
            self.humanoid_limb_and_weights = []
            self.skeleton_trees = []
            robot_cfg = {
                "mesh": self._has_mesh,
                "replace_feet": self._replace_feet,
                "rel_joint_lm": self._has_jt_limit,
                "upright_start": self._has_upright_start,
                "remove_toe": self.remove_toe,
                "real_weight_porpotion": self._real_weight_porpotion,
                "real_weight": self._real_weight,
                "masterfoot": self._masterfoot,
                "master_range": self._master_range,
                "big_ankle": self.big_ankle,
                "model": "smpl",
                "body_params": {},
                "joint_params": {},
                "geom_params": {},
                "actuator_params": {},
            }
            smpl_robot = Robot(
                robot_cfg,
                data_dir="data/smpl",
            )

            robot_cfg['masterfoot'] = False
            smpl_robot_orig = Robot(
                robot_cfg,
                data_dir="data/smpl",
            )

            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

            if self.has_shape_variation:
                #multiprocessing.set_sharing_strategy('file_system')
                queue = multiprocessing.Queue()
                num_jobs = 1
                #if num_jobs < 8: num_jobs = 1
                if flags.debug:
                    num_jobs = 1
    
                #num_jobs = 1
                res_acc = {}
                jobs = np.arange(num_envs)
                chunk = np.ceil(len(jobs) / num_jobs).astype(int)
                jobs = [
                    jobs[i:i + chunk] for i in range(0, len(jobs), chunk)
                ]
                job_args = [jobs[i] for i in range(len(jobs))]
                print(f'Using {num_jobs} for building agents!!!')
                for i in range(1, len(jobs)):
                    worker_args = (job_args[i], smpl_robot, queue, i)
                    worker = multiprocessing.Process(
                        target=self._create_smpl_humanoid_xml,
                        args=worker_args)
                    worker.start()
                res_acc.update(self._create_smpl_humanoid_xml(jobs[0], smpl_robot, None, 0))
                for i in tqdm(range(len(jobs) - 1)):
                    res = queue.get()
                    res_acc.update(res)
                    

                if flags.debug:
                    asset_options.fix_base_link = True

                for idx in np.arange(num_envs):
                    gender_beta, asset_file_real = res_acc[idx]
                    self.humanoid_files[idx] = asset_file_real
                    humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file_real, asset_options)
                    actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
                    motor_efforts = [prop.motor_effort for prop in actuator_props]
                    sk_tree = SkeletonTree.from_mjcf(asset_file_real)

                    # create force sensors at the feet
                    right_foot_idx = self.gym.find_asset_rigid_body_index( humanoid_asset, "L_Ankle")
                    left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "R_Ankle")
                    sensor_pose = gymapi.Transform()

                    self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx,sensor_pose)
                    self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx,sensor_pose)

                    self.humanoid_betas.append(torch.from_numpy(gender_beta).float())
                    self.humanoid_assets.append(humanoid_asset)
                    self.skeleton_trees.append(sk_tree)

                # joblib.dump(self.humanoid_betas, "humanoid_betas.pkl")
                smpl_robot.remove_geoms()  # Clean up the geoms
                self.humanoid_betas = torch.vstack(self.humanoid_betas).to(self.device)
            else:
                gender_beta, asset_file_real = self._create_smpl_humanoid_xml([0], smpl_robot, None, 0)[0]
                sk_tree = SkeletonTree.from_mjcf(asset_file_real)

                humanoid_asset = self.gym.load_asset(self.sim, asset_root,
                                                     asset_file_real,
                                                     asset_options)
                actuator_props = self.gym.get_asset_actuator_properties(
                    humanoid_asset)
                motor_efforts = [prop.motor_effort for prop in actuator_props]

                # create force sensors at the feet
                right_foot_idx = self.gym.find_asset_rigid_body_index(
                    humanoid_asset, "right_foot")
                left_foot_idx = self.gym.find_asset_rigid_body_index(
                    humanoid_asset, "left_foot")
                sensor_pose = gymapi.Transform()

                self.gym.create_asset_force_sensor(humanoid_asset,
                                                   right_foot_idx, sensor_pose)
                self.gym.create_asset_force_sensor(humanoid_asset,
                                                   left_foot_idx, sensor_pose)
                self.humanoid_betas = torch.tensor(np.array([gender_beta] * num_envs)).float().to(self.device)
                self.humanoid_assets = [humanoid_asset] * num_envs
                self.skeleton_trees = [sk_tree] * num_envs
        else:

            asset_path = os.path.join(asset_root, asset_file)
            asset_root = os.path.dirname(asset_path)
            asset_file = os.path.basename(asset_path)

            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            #asset_options.fix_base_link = True
            humanoid_asset = self.gym.load_asset(self.sim, asset_root,
                                                 asset_file, asset_options)

            actuator_props = self.gym.get_asset_actuator_properties(
                humanoid_asset)
            motor_efforts = [prop.motor_effort for prop in actuator_props]

            # create force sensors at the feet
            right_foot_idx = self.gym.find_asset_rigid_body_index(
                humanoid_asset, "right_foot")
            left_foot_idx = self.gym.find_asset_rigid_body_index(
                humanoid_asset, "left_foot")
            sensor_pose = gymapi.Transform()

            self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx,
                                               sensor_pose)
            self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx,
                                               sensor_pose)
            self.humanoid_assets = [humanoid_asset] * num_envs

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)
        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)
        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        self._setup_humanoid_misc(self.humanoid_files)

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, self.humanoid_assets[i])
            self.envs.append(env_ptr)
        self.humanoid_limb_and_weights = torch.stack(self.humanoid_limb_and_weights).to(self.device)
        print("Humanoid Weights", self.humanoid_masses[:10])
        # default_pose_scene = gymapi.Transform()
        # default_pose_scene.p = gymapi.Vec3(0,0,1)
        # self.gym.create_actor(env_ptr, self._scene_asset, default_pose_scene, 'Stairs', 0, 0, 0)


        dof_prop = self.gym.get_actor_dof_properties(self.envs[0],
                                                     self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower,
                                         device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper,
                                         device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()


        return

    def _setup_humanoid_misc(self, humanoid_files):
        pass

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        if self._divide_group or flags.divide_group:
            col_group = self._group_ids[env_id]
        else:
            col_group = env_id  # no inter-environment collision

        col_filter = 0
        if self.smpl_humanoid and (not self._has_self_collision):
            col_filter = 1

        start_pose = gymapi.Transform()
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        if (asset_file == "mjcf/ov_humanoid.xml"
                or asset_file == "mjcf/ov_humanoid_sword_shield.xml"):
            char_h = 0.927
        else:
            char_h = 0.89

        pos = torch.tensor(get_axis_params(char_h,
                                           self.up_axis_idx)).to(self.device)
        pos[:2] += torch_rand_float(
            -1., 1., (2, 1), device=self.device).squeeze(
                1)  # ZL: segfault if we do not randomize the position

        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter, 0)
        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)
        mass_ind = [
            prop.mass for prop in self.gym.get_actor_rigid_body_properties(
                env_ptr, humanoid_handle)
        ]
        humanoid_mass = np.sum(mass_ind)
        self.humanoid_masses.append(humanoid_mass)

        curr_skeleton_tree = self.skeleton_trees[env_id]
        limb_lengths = torch.norm(curr_skeleton_tree.local_translation, dim=-1)
        masses = torch.tensor(mass_ind)


        # humanoid_limb_weight = torch.cat([limb_lengths[1:], masses])

        limb_lengths = [limb_lengths[group].sum() for group in self.limb_weight_group]
        masses = [masses[group].sum() for group in self.limb_weight_group]
        humanoid_limb_weight = torch.tensor(limb_lengths + masses)
        self.humanoid_limb_and_weights.append(humanoid_limb_weight) # ZL: attach limb lengths and full body weight.

        if self.smpl_humanoid:
            gender = self.humanoid_betas[env_id, 0].long()
            if gender == 0: gender = 1
            elif gender == 1: gender = 2
            elif gender == 2: gender = 0

            # color = torch.zeros(3)
            # color[gender] = 1 - np.clip((humanoid_mass - 70) / 70, 0,1)

            color = torch.tensor(agt_color(env_id))

            color_vec = gymapi.Vec3(*color)
        else:
            color_vec = gymapi.Vec3(0.54, 0.85, 0.2)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j,
                                          gymapi.MESH_VISUAL, color_vec)

        dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
        if (self._pd_control):
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            if self.has_shape_variation:
                pd_scale = humanoid_mass / self.cfg['env'].get('default_humanoid_mass', 77.0 if self._real_weight else 35.0)
                dof_prop['stiffness'] *= pd_scale * self._kp_scale
                dof_prop['damping'] *= pd_scale * self._kd_scale

        else:
            dof_prop["driveMode"] = gymapi.DOF_MODE_EFFORT
        self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)


        if self.smpl_humanoid and self._has_self_collision:
            # compliance_vals = [0.1] * 24
            # thickness_vals = [1.0] * 24
            if self._has_mesh:
                if self._masterfoot:
                    filter_ints = [0, 1, 96, 192, 192, 192, 192, 192, 192, 192, 1, 384, 288, 288, 288, 288, 288, 288, 288, 1024, 6, 1560, 0, 512, 0, 20, 0, 0, 0, 0, 10, 0, 0, 0]
                else:
                    filter_ints = [0, 1, 224, 512, 384, 1, 1792, 64, 1056, 4096, 6, 6168, 0, 2048, 0, 20, 0, 0, 0, 0, 10, 0, 0, 0]
            else:
                if self._masterfoot:
                    # filter_ints = [0, 0, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 12, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 32, 0, 48, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    filter_ints = [0, 0, 3, 6, 6, 6, 6, 6, 6, 6, 0, 12, 9, 9, 9, 9, 9, 9, 9, 32, 0, 48, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                else:
                    filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, humanoid_handle)

            assert(len(filter_ints) == len(props))
            for p_idx in range(len(props)):
                props[p_idx].filter = filter_ints[p_idx]
                # props[p_idx].friction = 50
                # props[p_idx].rolling_friction = 50
                # props[p_idx].torsion_friction
                # = 5
                # props[p_idx].compliance = compliance_vals[p_idx]
                # props[p_idx].thickness = thickness_vals[p_idx]

            self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)

        self.humanoid_handles.append(humanoid_handle)

        return

    def _build_pd_action_offset_scale(self):
        num_joints = len(self._dof_offsets) - 1

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]
            if not self._bias_offset:
                if (dof_size == 3):
                    curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                    curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                    curr_low = np.max(np.abs(curr_low))
                    curr_high = np.max(np.abs(curr_high))
                    curr_scale = max([curr_low, curr_high])
                    curr_scale = 1.2 * curr_scale
                    curr_scale = min([curr_scale, np.pi])

                    lim_low[dof_offset:(dof_offset + dof_size)] = -curr_scale
                    lim_high[dof_offset:(dof_offset + dof_size)] = curr_scale

                    #lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                    #lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

                elif (dof_size == 1):
                    curr_low = lim_low[dof_offset]
                    curr_high = lim_high[dof_offset]
                    curr_mid = 0.5 * (curr_high + curr_low)

                    # extend the action range to be a bit beyond the joint limits so that the motors
                    # don't lose their strength as they approach the joint limits
                    curr_scale = 0.7 * (curr_high - curr_low)
                    curr_low = curr_mid - curr_scale
                    curr_high = curr_mid + curr_scale

                    lim_low[dof_offset] = curr_low
                    lim_high[dof_offset] = curr_high
            else:
                curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                curr_mid = 0.5 * (curr_high + curr_low)

                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset:(dof_offset + dof_size)] = curr_low
                lim_high[dof_offset:(dof_offset + dof_size)] = curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset,
                                          device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale,
                                         device=self.device)
        if self.smpl_humanoid:
            self._L_knee_dof_idx = self._dof_names.index("L_Knee") * 3 + 1
            self._R_knee_dof_idx = self._dof_names.index("R_Knee") * 3 + 1
            # ZL: Modified SMPL
            self._pd_action_scale[self._L_knee_dof_idx] = 5
            self._pd_action_scale[self._R_knee_dof_idx] = 5

            if self._has_smpl_pd_offset:
                if self._has_upright_start:
                    self._pd_action_offset[self._dof_names.index("L_Shoulder") * 3] = -np.pi / 2
                    self._pd_action_offset[self._dof_names.index("R_Shoulder") * 3] = np.pi / 2
                else:
                    self._pd_action_offset[self._dof_names.index("L_Shoulder") * 3] = -np.pi / 6
                    self._pd_action_offset[self._dof_names.index("L_Shoulder") * 3 + 2] = -np.pi / 2
                    self._pd_action_offset[self._dof_names.index("R_Shoulder") * 3] = -np.pi / 3
                    self._pd_action_offset[self._dof_names.index("R_Shoulder") * 3 + 2] = np.pi / 2

        return

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf, self.progress_buf, self._contact_forces,
            self._contact_body_ids, self._rigid_body_pos,
            self.max_episode_length, self._enable_early_termination,
            self._termination_heights)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)

        if self.has_flip_observation:
            flip_obs = self._compute_flip_humanoid_obs(env_ids)
            if (env_ids is None):
                self._flip_obs_buf[:] = flip_obs
            else:
                self._flip_obs_buf[env_ids] = flip_obs

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_flip_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos.clone()
            body_rot = self._rigid_body_rot.clone()
            body_vel = self._rigid_body_vel.clone()
            body_ang_vel = self._rigid_body_ang_vel.clone()
        else:
            body_pos = self._rigid_body_pos[env_ids].clone()
            body_rot = self._rigid_body_rot[env_ids].clone()
            body_vel = self._rigid_body_vel[env_ids].clone()
            body_ang_vel = self._rigid_body_ang_vel[env_ids].clone()

        body_pos[..., 1] *= -1 # position
        body_pos = body_pos[..., self.left_to_right_index, :]

        body_rot[..., 0] *= -1 # angular rotation, global
        body_rot[..., 2] *= -1
        body_rot = body_rot[..., self.left_to_right_index, :]

        body_vel[..., 1] *= -1 # linear velocity
        body_vel = body_vel[..., self.left_to_right_index, :]

        body_ang_vel[..., 0] *= -1 # angular velocity
        body_ang_vel[..., 2] *= -1
        body_ang_vel = body_ang_vel[..., self.left_to_right_index, :]

        if self.smpl_humanoid:
            if (env_ids is None):
                smpl_params = self.humanoid_betas
                limb_weights = self.humanoid_limb_and_weights
            else:
                smpl_params = self.humanoid_betas[env_ids]
                limb_weights = self.humanoid_limb_and_weights[env_ids]

            obs = compute_humanoid_observations_smpl_max(
                body_pos, body_rot, body_vel, body_ang_vel, smpl_params, limb_weights,
                self._local_root_obs, self._root_height_obs,
                self._has_upright_start, self._has_shape_obs, self._has_limb_weight_obs)


        del body_pos, body_rot, body_vel, body_ang_vel
        return obs

    def _compute_humanoid_obs(self, env_ids=None):
        if (ENABLE_MAX_COORD_OBS):
            if (env_ids is None):
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
            else:
                body_pos = self._rigid_body_pos[env_ids]
                body_rot = self._rigid_body_rot[env_ids]
                body_vel = self._rigid_body_vel[env_ids]
                body_ang_vel = self._rigid_body_ang_vel[env_ids]
            if self.smpl_humanoid:
                if (env_ids is None):
                    smpl_params = self.humanoid_betas
                    limb_weights = self.humanoid_limb_and_weights
                else:
                    smpl_params = self.humanoid_betas[env_ids]
                    limb_weights = self.humanoid_limb_and_weights[env_ids]

                obs = compute_humanoid_observations_smpl_max(
                    body_pos, body_rot, body_vel, body_ang_vel, smpl_params,limb_weights,
                    self._local_root_obs, self._root_height_obs,
                    self._has_upright_start, self._has_shape_obs, self._has_limb_weight_obs)

            else:
                obs = compute_humanoid_observations_max(
                    body_pos, body_rot, body_vel, body_ang_vel,
                    self._local_root_obs, self._root_height_obs)

        else:
            if (env_ids is None):
                root_pos = self._rigid_body_pos[:, 0, :]
                root_rot = self._rigid_body_rot[:, 0, :]
                root_vel = self._rigid_body_vel[:, 0, :]
                root_ang_vel = self._rigid_body_ang_vel[:, 0, :]
                dof_pos = self._dof_pos
                dof_vel = self._dof_vel
                key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            else:
                root_pos = self._rigid_body_pos[env_ids][:, 0, :]
                root_rot = self._rigid_body_rot[env_ids][:, 0, :]
                root_vel = self._rigid_body_vel[env_ids][:, 0, :]
                root_ang_vel = self._rigid_body_ang_vel[env_ids][:, 0, :]
                dof_pos = self._dof_pos[env_ids]
                dof_vel = self._dof_vel[env_ids]
                key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]

            if self.smpl_humanoid and self.self.has_shape_obs:
                if (env_ids is None):
                    smpl_params = self.humanoid_betas
                else:
                    smpl_params = self.humanoid_betas[env_ids]
                obs = compute_humanoid_observations_smpl(
                    root_pos, root_rot, root_vel, root_ang_vel, dof_pos,
                    dof_vel, key_body_pos, self._dof_obs_size,
                    self._dof_offsets, smpl_params, self._local_root_obs,
                    self._root_height_obs, self._has_upright_start,
                    self._has_shape_obs)
            else:
                obs = compute_humanoid_observations(
                    root_pos, root_rot, root_vel, root_ang_vel, dof_pos,
                    dof_vel, key_body_pos, self._local_root_obs,
                    self._root_height_obs, self._dof_obs_size,
                    self._dof_offsets)
        return obs

    def _reset_actors(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        return

    def pre_physics_step(self, actions):
        
        self.actions = actions.to(self.device).clone()
        dof_actions = self.actions[:, :self.num_dof]

        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(dof_actions)

            if self.smpl_humanoid :
                pd_tar[:, self._dof_names.index("L_Hand") * 3:(self._dof_names.index("L_Hand") * 3 + 3)] = 0
                pd_tar[:, self._dof_names.index("R_Hand") * 3:(self._dof_names.index("R_Hand") * 3 + 3)] = 0

                if self._freeze_toe:
                    pd_tar[:, self._dof_names.index("L_Toe") *3:(self._dof_names.index("L_Toe") * 3 + 3)] = 0
                    pd_tar[:, self._dof_names.index("R_Toe") *3:(self._dof_names.index("R_Toe") * 3 + 3)] = 0
                if self._remove_neck:
                    pd_tar[:, self._dof_names.index("Neck") * 3:(self._dof_names.index("Neck") * 3 + 3)] = 0
                    pd_tar[:, self._dof_names.index("Head") * 3:(self._dof_names.index("Head") * 3 + 3)] = 0

            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = dof_actions * self.motor_efforts.unsqueeze(
                0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        
        return

    def post_physics_step(self):
        if not self.paused:
            self.progress_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()


        self.extras["terminate"] = self._terminate_buf
        self.extras["reward_raw"] = self.reward_raw.detach()

        if self.has_flip_observation:
            self.extras['flip_obs'] = self._flip_obs_buf
            self.extras['obs'] = self.obs_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self, sync_frame_time=False):
        if self.viewer:
            self._update_camera()

        super().render(sync_frame_time)
        return

    def _build_key_body_ids_tensor(self, key_body_names):
        if self.smpl_humanoid:
            body_ids = [self._body_names.index(name) for name in key_body_names]
            body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)

        else:
            env_ptr = self.envs[0]
            actor_handle = self.humanoid_handles[0]
            body_ids = []

            for body_name in key_body_names:
                body_id = self.gym.find_actor_rigid_body_handle(
                    env_ptr, actor_handle, body_name)
                assert (body_id != -1)
                body_ids.append(body_id)

            body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)


        return body_ids

    def _build_key_body_ids_orig_tensor(self, key_body_names):
        body_ids = [self._body_names_orig.index(name) for name in key_body_names]
        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[
            0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0],
                              self._cam_prev_char_pos[1] - 3.0, 1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0],
                                  char_root_pos[1] + cam_delta[1], cam_pos[2])

        self.gym.set_camera_location(self.recorder_camera_handle, self.envs[0],
                                     new_cam_pos, new_cam_target)
        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos,
                                       new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def dof_to_obs_smpl(pose):
    # type: (Tensor) -> Tensor
    joint_obs_size = 6
    B, jts = pose.shape
    num_joints = int(jts / 3)

    joint_dof_obs = torch_utils.quat_to_tan_norm(
        torch_utils.exp_map_to_quat(pose.reshape(-1, 3))).reshape(B, -1)
    assert ((num_joints * joint_obs_size) == joint_dof_obs.shape[1])

    return joint_dof_obs


@torch.jit.script
def dof_to_obs(pose, dof_obs_size, dof_offsets):
    # ZL this can be sped up for SMPL
    # type: (Tensor, int, List[int]) -> Tensor
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size, )
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
            axis = torch.tensor([0.0, 1.0, 0.0],
                                dtype=joint_pose.dtype,
                                device=pose.device)
            joint_pose_q = quat_from_angle_axis(joint_pose[..., 0], axis)
        else:
            joint_pose_q = None
            assert (False), "Unsupported joint type"

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        dof_obs[:, (j * joint_obs_size):((j + 1) *
                                         joint_obs_size)] = joint_dof_obs

    assert ((num_joints * joint_obs_size) == dof_obs_size)

    return dof_obs


@torch.jit.script
def compute_humanoid_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                  dof_pos, dof_vel, key_body_pos,
                                  local_root_obs, root_height_obs,
                                  dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    local_root_vel = torch_utils.my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = torch_utils.my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat(
        (1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2])
    local_end_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)

    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel,
                     local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos),
                    dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel,
                                      body_ang_vel, local_root_obs,
                                      root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1],
        local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot,
                                                     flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0],
        local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1],
                                     body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0],
        body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1],
                                     body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot,
                                                     flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(
        body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(
        body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(
        flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs,
                     local_body_vel, local_body_ang_vel),
                    dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf,
                           contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination,
                           termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        # if fall_contact.any():
        # print(masked_contact_buf[0, :, 0].nonzero())
        #     import ipdb
        #     ipdb.set_trace()


        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        ############################## Debug ##############################
        # mujoco_joint_names = np.array(['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']);  print( mujoco_joint_names[masked_contact_buf[0, :, 0].nonzero().cpu().numpy()])
        ############################## Debug ##############################

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)


    reset = torch.where(progress_buf >= max_episode_length - 1,
                        torch.ones_like(reset_buf), terminated)
    # import ipdb
    # ipdb.set_trace()

    return reset, terminated


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def remove_base_rot(quat):
    # ZL: removing the base rotation for SMPL model
    base_rot = quat_conjugate(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat))
    shape = quat.shape[0]
    return quat_mul(quat, base_rot.repeat(shape, 1))
    # return quat


@torch.jit.script
def compute_humanoid_observations_smpl(root_pos, root_rot, root_vel,
                                       root_ang_vel, dof_pos, dof_vel,
                                       key_body_pos, dof_obs_size, dof_offsets,
                                       smpl_params, local_root_obs,
                                       root_height_obs, upright,
                                       has_smpl_params):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, List[int], Tensor, bool, bool,bool, bool) -> Tensor
    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    local_root_vel = torch_utils.my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = torch_utils.my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat( (1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2])
    local_end_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)

    obs_list = []
    if root_height_obs: obs_list.append(root_h_obs)
    obs_list += [
        root_rot_obs,
        local_root_vel,
        local_root_ang_vel,
        dof_obs,
        dof_vel,
        flat_local_key_pos,
    ]
    if has_smpl_params: obs_list.append(smpl_params)
    obs = torch.cat(obs_list, dim=-1)

    return obs


@torch.jit.script
def compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel,
                                           body_ang_vel, smpl_params, limb_weight_params,
                                           local_root_obs, root_height_obs,
                                           upright, has_smpl_params, has_limb_weight_params):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1],
        local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot,
                                                     flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0],
        local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1],
                                     body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0],
        body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(
        body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(
        body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(
        flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    obs_list = []
    if root_height_obs: obs_list.append(root_h_obs)
    obs_list += [
        local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel
    ]
    if has_smpl_params: obs_list.append(smpl_params[:, :-6])
    if has_limb_weight_params: obs_list.append(limb_weight_params)

    obs = torch.cat(obs_list, dim=-1)
    return obs
