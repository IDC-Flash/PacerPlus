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

PERTURB_OBJS = [ ["small", 60],]

class Humanoid(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id,
                 headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.has_task = False
        self.headless = headless

        self.load_robot_configs(cfg)

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
        self._state_reset_happened = False

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
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        # refresh the tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)


        # obtain the 
        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()

        self._humanoid_root_states = self._root_states.view(
            self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]

        self._humanoid_actor_ids = num_actors * torch.arange(
            self.num_envs, device=self.device, dtype=torch.int32)
        
        self._base_quat = self._humanoid_root_states[:self.num_envs, 3:7]
        self._base_pos = self._humanoid_root_states[:self.num_envs, 0:3]

        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self.dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dofs, 0]
        self.dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dofs, 1]


        net_contact_forces_tensor = gymtorch.wrap_tensor(net_contact_forces)
        self._contact_forces = net_contact_forces_tensor.reshape(self.num_envs, -1, 3)


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

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
            self.num_envs, self.num_dofs)
        
        self._terminate_buf = torch.ones(self.num_envs,
                                         device=self.device,
                                         dtype=torch.long)

        self._build_termination_heights()

        if self.viewer != None:
            self._init_camera()

        ###################################
        # if self.has_flip_observation:
        #     self._flip_obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)

        
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.4,         
           'left_knee_joint' : 0.8,       
           'left_ankle_joint' : -0.4,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.4,                                       
           'right_knee_joint' : 0.8,                                             
           'right_ankle_joint' : -0.4,                                     
           'torso_joint' : 0., 
           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
        }
        # joint positions offsets and PD gains
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = default_joint_angles[name]
            self.default_dof_pos[i] = angle

            found = False
            for dof_name in self.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.stiffness[dof_name]
                    self.d_gains[i] = self.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def load_robot_configs(self, cfg):
        self._divide_group = cfg["env"].get("divide_group", False)
        self._group_obs = cfg["env"].get("group_obs", False)
        self._disable_group_obs = cfg["env"].get("disable_group_obs", False)
        if self._divide_group:
            self._group_num_people = group_num_people = min(cfg['env'].get("num_env_group", 128), cfg['env']['numEnvs'])
            self._group_ids = torch.tensor(np.arange(cfg["env"]["numEnvs"] / group_num_people).repeat(group_num_people).astype(int))

        self.motion_sym_loss = cfg["env"].get("motion_sym_loss", False)
        self.has_flip_observation = cfg['env'].get("hasFlipObservation", False)
        if self.motion_sym_loss:
            self.has_flip_observation = True

        self.hard_negative = cfg["env"].get("hard_negative", False) # hard negative sampling for im
        self.cycle_motion = cfg["env"].get("cycle_motion", False)  # Cycle motion to reach 300

        self._body_names_orig = ['pelvis',
                                'left_hip_yaw_link',
                                'left_hip_roll_link',
                                'left_hip_pitch_link',
                                'left_knee_link',
                                'left_ankle_link',
                                'right_hip_yaw_link',
                                'right_hip_roll_link',
                                'right_hip_pitch_link',
                                'right_knee_link',
                                'right_ankle_link',
                                'torso_link',
                                'left_shoulder_pitch_link',
                                'left_shoulder_roll_link',
                                'left_shoulder_yaw_link',
                                'left_elbow_link',
                                'right_shoulder_pitch_link',
                                'right_shoulder_roll_link',
                                'right_shoulder_yaw_link',
                                'right_elbow_link']
        _body_names_orig_copy = self._body_names_orig.copy()
        self._full_track_bodies = _body_names_orig_copy
        self._upper_track_bodies = ['pelvis',
                                    'torso_link',                                 
                                    'left_shoulder_pitch_link',
                                    'left_shoulder_roll_link',
                                    'left_shoulder_yaw_link',
                                    'left_elbow_link',
                                    'right_shoulder_pitch_link',
                                    'right_shoulder_roll_link',
                                    'right_shoulder_yaw_link',
                                    'right_elbow_link']


        self._body_names = self._body_names_orig
        self._dof_names = self._body_names[1:]
 

        self.locomotion_reward_scales = {
            'lin_vel_z': 0.0,
            'ang_vel_xy': 0.0,
            'orientation': -1.0,
            'torques': -0.00001,
            'dof_acc':  -3.5e-8,
            'base_height': 0.0,
            'feet_air_time': 1.0,
            'collision': 0.0,
            'action_rate': -0.01,
            'dof_pos_limits': -10.0}
        self.num_locomotion_reward = 0
        for key in self.locomotion_reward_scales.keys():
            if self.locomotion_reward_scales[key] != 0:
                self.num_locomotion_reward += 1


    def _clear_recorded_states(self):
        del self.state_record
        self.state_record = defaultdict(list)

    def _record_states(self):
        self.state_record['dof_pos'].append(self.dof_pos.cpu().clone())
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
        fps = 30
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
            self._state_reset_happened = True
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

        self.actions[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self._contact_forces[env_ids] = 0
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _setup_character_props(self, key_bodies):
        self._dof_body_ids = np.arange(1, len(self._body_names))
        self._dof_offsets = np.linspace(0, len(self._dof_names), len(self._body_names)).astype(int)
        self._num_actions = 19
        self._num_self_obs = 66
        return

    def _build_termination_heights(self):
        head_term_height = 0.3
        termination_height = self.cfg["env"]["terminationHeight"]
        self._termination_heights = np.array([termination_height] *
                                             self.num_bodies)

        head_id = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles[0], "head")
        self._termination_heights[head_id] = max(
            head_term_height, self._termination_heights[head_id])

        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        self._termination_heights = to_torch(self._termination_heights,
                                             device=self.device)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.getcwd()
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        # assets from legged gyms
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = 3
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.fix_base_link = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0
        asset_options.linear_damping = 0
        asset_options.max_angular_velocity =  1000.
        asset_options.max_linear_velocity =  1000.
        asset_options.armature = 0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False


        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle': 40,
                     'torso': 300,
                     'shoulder': 100,
                     "elbow":100,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 6,
                     'ankle': 2,
                     'torso': 6,
                     'shoulder': 2,
                     "elbow":2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        self.stiffness = stiffness
        self.damping = damping

        for i in range(self.num_dofs):
            dof_props_asset['driveMode'][i] = gymapi.DOF_MODE_POS
            # joint positions offsets and PD gains
            name = self.dof_names[i]
            for dof_name in self.stiffness.keys():
                if dof_name in name:
                    dof_props_asset['stiffness'][i] = self.stiffness[dof_name]
                    dof_props_asset['damping'][i] = self.damping[dof_name]


        # get the names of the feet and the contact bodies  
        foot_name = 'ankle'        
        feet_names = [s for s in body_names if foot_name in s]
        penalized_contact_names = []
        for name in ["hip", "knee"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in ["pelvis"]:
            termination_contact_names.extend([s for s in body_names if name in s])


        actuator_props = self.gym.get_asset_actuator_properties(
            robot_asset)
        
        # sensor_options = gymapi.ForceSensorProperties()
        # sensor_options.enable_forward_dynamics_forces = False # for example gravity
        # sensor_options.enable_constraint_solver_forces = True # for example contacts
        # sensor_options.use_world_frame = True
        # # create force sensors at the feet
        # right_foot_idx = self.gym.find_asset_rigid_body_index(
        #     robot_asset, "right_ankle")
        # left_foot_idx = self.gym.find_asset_rigid_body_index(
        #     robot_asset, "left_ankle")
        # sensor_pose = gymapi.Transform()

        # self.gym.create_asset_force_sensor(robot_asset, right_foot_idx, sensor_pose, sensor_options)
        # self.gym.create_asset_force_sensor(robot_asset, left_foot_idx, sensor_pose,sensor_options)
        self.robot_assets = [robot_asset] * num_envs
 
        self.torso_index = 0
        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, self.robot_assets[i], dof_props_asset, rigid_shape_props_asset)
            self.envs.append(env_ptr)

        self._contact_body_ids = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            
            self._contact_body_ids[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], feet_names[i])

        self._penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self._penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], penalized_contact_names[i])

        self._termination_contact_body_ids = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self._termination_contact_body_ids[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], termination_contact_names[i])

        self.feet_air_time = torch.zeros(self.num_envs, self._contact_body_ids.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self._contact_body_ids), dtype=torch.bool, device=self.device, requires_grad=False)
        return

    ################ Callbacks for environment creation from LeggedGym################
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if not flags.test:
            if env_id==0:
                # prepare friction randomization
                friction_range = [0.5, 1.25]
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * 0.9
                self.dof_pos_limits[i, 1] = m + 0.5 * r * 0.9
            self._pd_action_offset = (self.dof_pos_limits[:, 1] + self.dof_pos_limits[:, 0]) / 2
            self._pd_action_scale = (self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]) / 2
        return props
    
    def _process_rigid_body_props(self, props, env_id):
        randomize_base_mass = False
        if randomize_base_mass:
            rng = [-1, 1]
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props


    def _build_env(self, env_id, env_ptr, humanoid_asset, dof_props_asset, rigid_shape_props_asset):
        if self._divide_group or flags.divide_group:
            col_group = self._group_ids[env_id]
        else:
            col_group = env_id  # no inter-environment collision

        col_filter = 0 # 1 for has self collision
        start_pose = gymapi.Transform()    
        char_h = 0.89
        pos = torch.tensor(get_axis_params(char_h, self.up_axis_idx)).to(self.device)
        pos[:2] += torch_rand_float( -1., 1., (2, 1), device=self.device).squeeze(1)  # ZL: segfault if we do not randomize the position

        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, env_id)
        self.gym.set_asset_rigid_shape_properties(humanoid_asset, rigid_shape_props)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "h1", col_group, col_filter, 0)
        dof_prop = self._process_dof_props(dof_props_asset, env_id)
        dof_prop["driveMode"] = gymapi.DOF_MODE_POS

        self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)
        body_props = self.gym.get_actor_rigid_body_properties(env_ptr, humanoid_handle)
        body_props = self._process_rigid_body_props(body_props, env_id)
        self.gym.set_actor_rigid_body_properties(env_ptr, humanoid_handle, body_props, recomputeInertia=True)


        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)
        color_vec = gymapi.Vec3(0.54, 0.85, 0.2)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j,
                                          gymapi.MESH_VISUAL, color_vec)

        self.humanoid_handles.append(humanoid_handle)

        return



    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
        return




    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf, self.progress_buf, self._contact_forces,
            self._termination_contact_body_ids, self._base_quat.clone(),
            self.max_episode_length, self._enable_early_termination)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        if self._state_reset_happened and "default_dof_pos" in self.__dict__:
            # ZL: Hack to get rigidbody pos and rot to be the correct values. Needs to be called after _set_env_state
            env_ids = self._reset_ref_env_ids
            if len(env_ids) > 0:
                self.dof_pos[env_ids] = self.default_dof_pos
                self._state_reset_happened = False
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)
        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
           self.obs_buf[env_ids] = obs

        return


    def _compute_humanoid_obs(self, env_ids=None):
        if env_ids is None:
            root_pos = self._base_pos.clone()
            root_rot = self._base_quat.clone()
            root_vel = self._humanoid_root_states[:, 7:10].clone()
            root_ang_vel = self._humanoid_root_states[:, 10:13].clone()
            dof_pos = self.dof_pos.clone()
            dof_vel = self.dof_vel.clone()
            action = self.actions.clone()
            gravity_vec = self.gravity_vec.clone()
        else:
            root_pos = self._base_pos[env_ids].clone()
            root_rot = self._base_quat[env_ids].clone()
            dof_pos = self.dof_pos[env_ids].clone()
            dof_vel = self.dof_vel[env_ids].clone()
            root_vel = self._humanoid_root_states[env_ids, 7:10].clone()
            root_ang_vel = self._humanoid_root_states[env_ids, 10:13].clone()
            action = self.actions[env_ids].clone()
            gravity_vec = self.gravity_vec[env_ids].clone()

        
        obs = compute_robot_observation(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, action, gravity_vec, self.default_dof_pos)

        return obs
    


    def _reset_actors(self, env_ids):
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        return
    

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        base_root_pos = torch.tensor([0, 0, 1]).to(self.device).unsqueeze(0).repeat((len(env_ids), 1)).float()
        base_root_rot = torch.tensor([0, 0, 0, 1]).to(self.device).unsqueeze(0).repeat((len(env_ids), 1)).float()
        self._humanoid_root_states[env_ids, 0:3] = base_root_pos
        self._humanoid_root_states[env_ids, 3:7] = base_root_rot
        # base velocities
        self._humanoid_root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def step(self, actions):

        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](
                actions)
            
        self.actions = actions

        self.render()

        for _ in range(self.control_freq_inv):
            self.pre_physics_step(self.actions)
            self.gym.simulate(self.sim)
            #self.gym.refresh_dof_state_tensor(self.sim)

            # to fix!
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step();

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations'][
                'noise_lambda'](self.obs_buf)


    def pre_physics_step(self, actions):
        #### Hz < 500 use PD control rather than torque control
        pd_tar = self.default_dof_pos + self._pd_action_scale * actions 
        pd_tar = torch.clamp(pd_tar, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
        pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
        self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
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

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        return

    def render(self, sync_frame_time=False):
        if self.viewer:
            self._update_camera()

        super().render(sync_frame_time)
        return


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
    

    #------------ basic reward functions from legged gyms----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self._root_states[:, 9])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self._root_states[:, 10:12]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        heading = torch_utils.calc_heading_quat_inv(self._humanoid_root_states[:, 3:7])
        projective_gravity = torch_utils.quat_rotate(heading, self.gravity_vec[:self.num_envs])
        return torch.sum(torch.square(projective_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._root_states[:, 2]
        return torch.square(base_height - 0.98)
    
    def _reward_torques(self):
        # Penalize torques
        torques = self.p_gains * (self.actions + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.sum(torch.square(torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self._contact_forces[:, self._penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        torques = self.p_gains * (self.actions + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.sum((torch.abs(torques) - self.torque_limits).clip(min=0.), dim=1)


    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self._contact_forces[:, self._contact_body_ids, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        #rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self._contact_forces[:, self._contact_body_ids, :2], dim=2) >\
             5 *torch.abs(self._contact_forces[:, self._contact_body_ids, 2]), dim=1)
        

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self._contact_forces[:, self._contact_body_ids, :], dim=-1) -  100).clip(min=0.), dim=1)
    

    def _build_locomotion_rewards(self):
        rewards = []
        for name, scale in self.locomotion_reward_scales.items():
            name = '_reward_' + name
            function = getattr(self, name)
            if scale != 0:
                rewards.append(function() * scale)
        return torch.stack(rewards, dim=1)

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
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

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

#@torch.jit.script
def compute_robot_observation(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, actions, gravity_vec, default_dof_pos, ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    erular_xyz = get_euler_xyz(root_rot)

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_root_vel = torch_utils.my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = torch_utils.my_quat_rotate(heading_rot, root_ang_vel)
    local_dof_pos = dof_pos - default_dof_pos
    projected_gravity = torch_utils.my_quat_rotate(heading_rot, gravity_vec)
    obs = torch.cat([local_root_vel, local_root_ang_vel, local_dof_pos, dof_vel, projected_gravity, actions], dim=-1)

    return obs



@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf,
                           termination_contact_body_ids, base_quat,
                           max_episode_length, enable_early_termination):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        # masked_contact_buf = contact_buf.clone()
        # masked_contact_buf[:, contact_body_ids, :] = 0
        # fall_contact = torch.any(torch.norm(masked_contact_buf, dim=-1) > 0.1, dim=1)
        fall_contact = torch.any(torch.norm(contact_buf[:,termination_contact_body_ids], dim=-1) > 1, dim=1)

        rpy = get_euler_xyz(base_quat)
        fall_rpy = torch.logical_or(torch.abs(rpy[:,1])>1.0, torch.abs(rpy[:,0])>0.8)

        has_fallen = torch.logical_or(fall_contact, fall_rpy)
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)


    reset = torch.where(progress_buf >= max_episode_length - 1,
                        torch.ones_like(reset_buf), terminated)
    # import ipdb
    # ipdb.set_trace()

    return reset, terminated




