import torch
import joblib
import os
import numpy as np

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import learning.common_player as common_player

class AMPPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._normalize_input = config['normalize_input']
        self._disc_reward_scale = config['disc_reward_scale']

        super().__init__(config)
        self.export_motion = self.task.cfg['args'].export_motion
        # self.env.task.update_value_func(self._eval_critic, self._eval_actor)
        humanoid_env = self.env.task
        self.export_motion = self.task.cfg['args'].export_motion
        if hasattr(humanoid_env,'terminate_dist'):
            humanoid_env.terminate_dist *= 2 # ZL Hack: use test 
        return

    def restore(self, fn):
        super().restore(fn)
        if self._normalize_amp_input:
            checkpoint = torch_ext.load_checkpoint(fn)
            self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])

            if self._normalize_input:
                self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        return

    def _build_net(self, config):
        super()._build_net(config)

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()

        return

    def _eval_critic(self, input):
        input = self._preproc_input(input)
        return self.model.a2c_network.eval_critic(input)

    def _post_step(self, info):
        super()._post_step(info)
        if (self.env.task.viewer):
            self._amp_debug(info)

        return

    def _eval_task_value(self, input):
        input = self._preproc_input(input)
        return self.model.a2c_network.eval_task_value(input)


    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
            if self.env.task.has_task:
                config['self_obs_size'] = self.env.task.get_self_obs_size()
                config['task_obs_size'] = self.env.task.get_task_obs_size()
                config['task_obs_size_detail'] = self.env.task.get_task_obs_size_detail()
                config['temporal_hist_length'] = self.env.task._temporal_hist_length ####### for observation 
                config['use_temporal_buf'] = self.env.task.use_temporal_buf
                config['temporal_buf_length'] = self.env.task._temporal_buf_length ####### for self observation, with out task
                config['has_flip_observation'] = self.task.has_flip_observation
                config['left_right_index'] = self.task.left_to_right_index_action
                config['amp_temporal_length'] = self.task._num_amp_obs_steps
                #config['use_trajectory_velocity'] = self.task.use_trajectory_velocity

        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
            if self.env.task.has_task:
                config['self_obs_size'] = self.vec_env.env.task.get_self_obs_size()
                config['task_obs_size'] = self.vec_env.env.task.get_task_obs_size()
                config['task_obs_size_detail'] = self.vec_env.env.task.get_task_obs_size_detail()
                config['temporal_hist_length'] = self.vec_env.env.task._temporal_hist_length ####### for observation 
                config['use_temporal_buf'] = self.vec_env.env.task.use_temporal_buf
                config['temporal_buf_length'] = self.vec_env.env.task._temporal_buf_length ####### for self observation, with out task
                config['has_flip_observation'] = self.vec_env.env.task.has_flip_observation
                config['left_right_index'] = self.vec_env.env.task.left_to_right_index_action
                config['amp_temporal_length'] = self.vec_env.env.task._num_amp_obs_steps
                #config['use_trajectory_velocity'] = self.vec_env.env.use_trajectory_velocity

        return config

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs_single = amp_obs[0:1]

            # left_to_right_index = [
            #     4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11, 12, 18, 19, 20, 21, 22,
            #     13, 14, 15, 16, 17
            # ]
            # action = self._eval_actor(info['obs'])[0]
            # flip_action = self._eval_actor(info['flip_obs'])[0]
            # flip_action = flip_action.view(-1, 23, 3)
            # flip_action[..., 0] *= -1
            # flip_action[..., 2] *= -1
            # flip_action[..., :] = flip_action[..., left_to_right_index, :]
            # print("flip diff", (flip_action.view(-1, 69) - action).norm(dim = 1))

            disc_pred = self._eval_disc(amp_obs_single)
            amp_rewards = self._calc_amp_rewards(amp_obs_single)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]

            # print("disc_pred: ", disc_pred, disc_reward)

            # if not "rewards" in self.__dict__:
            #     self.rewards = []
            # self.rewards.append(
            #     self._calc_amp_rewards(
            #         info['amp_obs'])['disc_rewards'].squeeze())
            # if len(self.rewards) > 500:
            #     print(torch.topk(torch.stack(self.rewards).mean(dim = 0), k=150, largest=False)[1])
            #     import ipdb; ipdb.set_trace()
            #     self.rewards = []

        # jp hack
        with torch.enable_grad():
            amp_obs_single = amp_obs[0:1]
            amp_obs_single.requires_grad_(True)
            disc_pred = self._eval_disc(amp_obs_single)

        disc_grad = torch.autograd.grad(
            disc_pred,
            amp_obs_single,
            grad_outputs=torch.ones_like(disc_pred),
            create_graph=False,
            retain_graph=True,
            only_inputs=True)
        grad_vals = torch.mean(torch.abs(disc_grad[0]), dim=0)
        if not "grad_acc" in self.__dict__:
            self.grad_acc = []
            self.reward_acc = []

        self.grad_acc.append(grad_vals)
        self.reward_acc.append(info['reward_raw'])
        if len(self.grad_acc) > 298:
            import joblib
            joblib.dump(self.grad_acc, "grad_acc.pkl")

            print(torch.stack(self.reward_acc).mean(dim = 0))
            self.grad_acc = []
            self.reward_acc = []
            # import ipdb; ipdb.set_trace()
            print("Dumping Grad info!!!!")

        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _eval_actor(self, input):
        input = self._preproc_input(input)
        return self.model.a2c_network.eval_actor(input)

    def _preproc_input(self, input):
        if self._normalize_input:
            input = self.running_mean_std(input)
        return input

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            for k,v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        if self.normalize_input:
            ######## process the output part
            if self.task.use_temporal_buf:
                temporal_buf_length = self.task._temporal_buf_length
                temporal_buf_size = self.vec_env.env.task.get_self_obs_size()
                obs_batch, temporal_buffer = obs_batch[:, :-temporal_buf_size * temporal_buf_length], obs_batch[:, -temporal_buf_size * temporal_buf_length:]

            if self.task._temporal_output:
               obs_batch = obs_batch.reshape(-1, obs_batch.shape[1] // self.task._temporal_hist_length, self.task._temporal_hist_length) 
            obs_batch = self.running_mean_std(obs_batch)
            obs_batch = obs_batch.reshape(obs_batch.shape[0], -1)
            if self.task.use_temporal_buf:
                obs_batch = torch.cat([obs_batch, temporal_buffer], dim=1)
            
        return obs_batch


    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_dumped = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None
        all_num_envs = self.env.task.num_envs
        if self.export_motion:
            dump_dict = {}
            for i in range(all_num_envs):
                dump_dict[i] = {}
                dump_dict[i]['done'] = 0
                dump_dict[i]['info'] = {}


        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for t in range(n_games):
            if has_dumped and self.export_motion:
                print('ALL DUMPED !!!')
                break

            if games_played >= n_games:
                break

            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            if self.export_motion:
                root_pos, root_rot, dof_pos, physics_kp, trajectory_target, \
                      imitation_target, d3_visible, tracking_mask, body_shape  = self.task.get_current_pose()
                root_pos_all = root_pos.unsqueeze(1)
                root_rot_all = root_rot.unsqueeze(1)
                dof_pos_all = dof_pos.unsqueeze(1)
                physics_kp = physics_kp.unsqueeze(1)
                trajectory_target = trajectory_target.unsqueeze(1)
                imitation_target = imitation_target.unsqueeze(1)
                d3_visible = d3_visible.unsqueeze(1)
                tracking_mask = tracking_mask.unsqueeze(1)
                for i in range(all_num_envs):
                    if  dump_dict[i]['done'] == 0:
                        dump_dict[i]['info']['root_pos_all'] = root_pos_all[i:i+1]
                        dump_dict[i]['info']['root_rot_all'] = root_rot_all[i:i+1]
                        dump_dict[i]['info']['dof_pos_all'] = dof_pos_all[i:i+1]
                        dump_dict[i]['info']['physics_kp_all'] = physics_kp[i:i+1]
                        dump_dict[i]['info']['body_shape'] = body_shape[i:i+1]
                        dump_dict[i]['info']['trajectory_target'] = trajectory_target[i:i+1]
                        dump_dict[i]['info']['imitation_target'] = imitation_target[i:i+1]
                        dump_dict[i]['info']['d3_visible'] = d3_visible[i:i+1]
                        dump_dict[i]['info']['tracking_mask'] = tracking_mask[i:i+1]


            with torch.no_grad():
                for n in range(self.max_steps):
                    obs_dict = self.env_reset(done_indices)
                    if has_masks:
                        masks = self.env.get_action_mask()
                        action = self.get_masked_action(obs_dict, masks, is_determenistic)
                    else:
                        action = self.get_action(obs_dict, is_determenistic)
                        
                    obs_dict, r, done, info =  self.env_step(self.env, action)

                    cr += r
                    steps += 1

                    self._post_step(info)

                    if render:
                        self.env.render(mode = 'human')
                        time.sleep(self.render_sleep)

                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[::self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if self.export_motion:
                        root_pos, root_rot, dof_pos, physics_kp, trajectory_target, \
                            imitation_target, d3_visible, tracking_mask, body_shape  = self.task.get_current_pose()
                        for i in range(all_num_envs):
                            if  dump_dict[i]['done'] == 0:
                                dump_dict[i]['info']['root_pos_all'] = torch.cat([dump_dict[i]['info']['root_pos_all'], root_pos.unsqueeze(1)[i:i+1]], dim=1)
                                dump_dict[i]['info']['root_rot_all'] = torch.cat([dump_dict[i]['info']['root_rot_all'], root_rot.unsqueeze(1)[i:i+1]], dim=1)
                                dump_dict[i]['info']['dof_pos_all'] = torch.cat([dump_dict[i]['info']['dof_pos_all'], dof_pos.unsqueeze(1)[i:i+1]], dim=1)
                                dump_dict[i]['info']['physics_kp_all'] = torch.cat([dump_dict[i]['info']['physics_kp_all'], physics_kp.unsqueeze(1)[i:i+1]], dim=1)
                                dump_dict[i]['info']['trajectory_target'] = torch.cat([dump_dict[i]['info']['trajectory_target'], trajectory_target.unsqueeze(1)[i:i+1]], dim=1)
                                dump_dict[i]['info']['imitation_target'] = torch.cat([dump_dict[i]['info']['imitation_target'], imitation_target.unsqueeze(1)[i:i+1]], dim=1)
                                dump_dict[i]['info']['d3_visible'] = torch.cat([dump_dict[i]['info']['d3_visible'], d3_visible.unsqueeze(1)[i:i+1]], dim=1)
                                dump_dict[i]['info']['tracking_mask'] = torch.cat([dump_dict[i]['info']['tracking_mask'], tracking_mask.unsqueeze(1)[i:i+1]], dim=1)

                    
                        done_envs = 0
                        for i in range(all_num_envs):
                            if done[i] == 1:
                                dump_dict[i]['done'] = 1
                            done_envs += dump_dict[i]['done']
                    
                        if done_envs == all_num_envs:
                            motion_export_path = self.task.cfg['env']['export_motion_path']
                            for i in range(all_num_envs):
                                for key in dump_dict[i]['info'].keys():
                                    dump_dict[i]['info'][key] = dump_dict[i]['info'][key].detach().cpu().numpy()
                                motion_export = dump_dict[i]['info']
                                motion_export = self._post_process_dump_data(i, motion_export)
                                os.makedirs(f'{motion_export_path}', exist_ok=True)
                                length = motion_export["root_pos_all"].shape[1]
                                print(f'Dump to: {motion_export_path}/{i}.pkl. Motion length: {length}.')
                                joblib.dump(motion_export, f'{motion_export_path}/{i}.pkl')
                            has_dumped = True
                            break

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if 'battle_won' in info:
                                print_game_res = True
                                game_res = info.get('battle_won', 0.5)
                            if 'scores' in info:
                                print_game_res = True
                                game_res = info.get('scores', 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)
                            else:
                                print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count)

                        sum_game_res += game_res
                        if batch_size//self.num_agents == 1 or games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]
                    

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)

        return


    def _post_process_dump_data(self, idx, motion_dump):
        skeleton_tree = self.task.skeleton_trees[idx]
        offset = skeleton_tree.local_translation[0]
        root_pos = motion_dump['root_pos_all'][0]
        root_rot = motion_dump['root_rot_all'][0]
        body_pos = motion_dump['dof_pos_all'][0]
        pose_aa = np.concatenate([root_rot[:, None, :], body_pos], axis=1)
        batch_size = pose_aa.shape[0]
        pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)[..., self.task.smpl_2_mujoco, :]
        root_trans_offset = root_pos - offset.numpy()
        sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            torch.from_numpy(root_pos),
            is_local=True)
        global_rot = sk_state.global_rotation
        B, J, N = global_rot.shape
        pose_quat_global = (sRot.from_quat(global_rot.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5])).as_quat().reshape(B, -1, 4)
        B_down = pose_quat_global.shape[0]
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat_global),
            torch.from_numpy(root_pos),
            is_local=False)
        local_rot = new_sk_state.local_rotation
        pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_rotvec().reshape(B_down, -1, 3)
        pose_aa = pose_aa[:, self.task.mujoco_2_smpl, :]
        motion_dump['root_rot_all'] = pose_aa[:, 0][None, ...]
        motion_dump['dof_pos_all'] = pose_aa[:, 1:][None, ...]
        return motion_dump