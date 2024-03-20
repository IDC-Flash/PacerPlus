# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import If
import numpy as np
import os
import yaml
from tqdm import tqdm
from amp.utils.flags import flags
from poselib.poselib.core.rotation3d import *
from isaacgym.torch_utils import *
from amp.utils import torch_utils
import joblib
import torch
import torch.multiprocessing as mp
import copy
import gc


USE_CACHE = True
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy

    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy



def load_motion_from_npz(ids, motion_data_list, queue, pid):
    # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
    res = {}
    for f in tqdm(range(len(motion_data_list))):
        assert (len(ids) == len(motion_data_list))
        curr_id = ids[f] # id for this datasample
        curr_file = motion_data_list[f]
        data = np.load(curr_file, allow_pickle=True)
        curr_motion = {}
        for key in data.keys():
            curr_motion[key] = torch.from_numpy(data[key])
        res[curr_id] = (curr_file, curr_motion)

    if not queue is None:
        queue.put(res)
    else:
        return res

class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                # print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1

        # print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out


class MotionLib():
    def __init__(self, motion_file, fps, device, debug=False):
        self.debug = debug
        self._device = device
        self._motion_data_list = [os.path.join(motion_file, f) for f in os.listdir(motion_file)]
        self._num_unique_motions = len(self._motion_data_list)
        self._fps = fps
        #### Termination history
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions)
        self._success_rate = torch.zeros(self._num_unique_motions)
        self._sampling_history = torch.zeros(self._num_unique_motions)
        self._sampling_prob = torch.ones(self._num_unique_motions)/self._num_unique_motions # For use in sampling batches
        self._sampling_batch_prob = None # For use in sampling within batches
        return

    def load_motions(self, num_envs, random_sample = True, start_idx = 0):
        # load motion load the same number of motions as there are skeletons (humanoids)
        if "gts" in self.__dict__:
            del self.gts , self.grs , self.lrs, self.grvs, self.gravs , self.gavs , self.gvs, self.dvs,
            del  self._motion_lengths, self._motion_fps, self._motion_dt, self._motion_num_frames, self._motion_bodies, self._motion_aa , self._motion_quat

        motions = []
        self._motion_lengths = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_bodies = []
        self._motion_aa = []
        self._motion_quat = []

        torch.cuda.empty_cache()
        gc.collect()

        total_len = 0.0
        num_motion_to_load = num_envs

        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True)
        else:
            sample_idxes = torch.clip(torch.arange(num_envs + start_idx, 0, self._num_unique_motions - 1))

        self._curr_motion_ids = sample_idxes

        self._sampling_batch_prob =  self._sampling_prob[self._curr_motion_ids]/self._sampling_prob[self._curr_motion_ids].sum()


        motion_data_list = [self._motion_data_list[i] for i in sample_idxes.cpu().int().numpy()]
        #mp.set_sharing_strategy('file_descriptor')

        manager = mp.Manager()
        queue = manager.Queue()
        num_jobs = 8
        #if num_jobs <= 8: num_jobs = 1
        # num_jobs = 1
        if self.debug:
            num_jobs = 1
        print("Using {} jobs loading reference motions".format(num_jobs))
        res_acc = {} # using dictionary ensures order of the results.
        jobs = motion_data_list
        chunk = np.ceil(len(jobs) / num_jobs).astype(int)
        ids = np.arange(len(jobs))

        jobs = [(ids[i:i + chunk], jobs[i:i + chunk]) for i in range(0, len(jobs), chunk)]
        job_args = [jobs[i] for i in range(len(jobs))]
        workers = []
        for i in range(1, len(jobs)):
            worker_args = (*job_args[i], queue, i)
            worker = mp.Process(target=load_motion_from_npz,
                                args=worker_args)
            worker.start()
            workers.append(worker)
        res_acc.update(load_motion_from_npz(*jobs[0], None, 0))

        for i in tqdm(range(len(jobs) - 1)):
            res = queue.get()
            res_acc.update(res)
        

        #res_acc = load_motion_with_skeleton(*jobs[0], None, 0)

        for f in tqdm(range(len(res_acc))):
            motion_file_data, curr_motion = res_acc[f]
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)

            motion_fps = self._fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.obj['base_pose'].shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            self._motion_lengths.append(curr_len)

        self._motion_lengths = torch.tensor(self._motion_lengths,
                                            device=self._device,
                                            dtype=torch.float32)
        self._motion_fps = torch.tensor(self._motion_fps,
                                        device=self._device,
                                        dtype=torch.float32)

        self._motion_dt = torch.tensor(self._motion_dt,
                                       device=self._device,
                                       dtype=torch.float32)
        
        self._motion_num_frames = torch.tensor(self._motion_num_frames,
                                               device=self._device)

        self._num_motions = len(motions)

        self.gts = torch.cat([torch.zeros_like(m.obj['base_pose']) for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.obj['base_pose'] for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.obj['base_velocity'] for m in motions], dim=0).float().to(self._device)
        self.gravs = torch.cat([m.obj['base_angular_velocity'] for m in motions], dim=0).float().to(self._device)
        self.lrs = torch.cat([m.obj['joint_position'] for m in motions], dim=0).float().to(self._device)
        self.lps = torch.cat([m.obj['link_location'] for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.obj['joint_velocity'] for m in motions], dim=0).float().to(self._device)
        self.lal = torch.cat([m.obj['left_ankle_location'] for m in motions], dim=0).float().to(self._device)
        self.ral = torch.cat([m.obj['right_ankle_location'] for m in motions], dim=0).float().to(self._device)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions),
                                       dtype=torch.long,
                                       device=self._device)
        motion = motions[0]
        self.num_bodies = motion.obj['joint_position'].shape[1] + 1

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        print("Loaded {:d} motions with a total length of {:.3f}s.".format(
            num_motions, total_len))
        return

    def num_motions(self):
        return self._num_motions

    def get_total_length(self):
        return sum(self._motion_lengths)

    def update_sampling_weight(self):
        sampling_temp = 0.2
        curr_termination_prob = 0.5
        curr_succ_rate = 1 - self._termination_history[self._curr_motion_ids]/ self._sampling_history[self._curr_motion_ids]
        self._success_rate[self._curr_motion_ids] = curr_succ_rate
        sample_prob = torch.exp(-self._success_rate/sampling_temp)

        self._sampling_prob =  sample_prob/sample_prob.sum()
        self._termination_history[self._curr_motion_ids] = 0
        self._sampling_history[self._curr_motion_ids] = 0

        topk_sampled = self._sampling_prob.topk(50)
        print("Current most sampled", self._motion_data_keys[topk_sampled.indices.cpu().numpy()])


    def update_sampling_history(self, env_ids):
        #from IPython import embed; embed()
        env_ids = env_ids.cpu()
        self._sampling_history[self._curr_motion_ids[env_ids]] += 1
        # print("sampling history: ", self._sampling_history[self._curr_motion_ids])

    def update_termination_history(self, termination):
        self._termination_history[self._curr_motion_ids] += termination.cpu()
        # print("termination history: ", self._termination_history[self._curr_motion_ids])


    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._sampling_batch_prob,
                                       num_samples=n,
                                       replacement=True).to(self._device)

        return motion_ids


    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def sample_time_interval(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time
        curr_fps = 1/30
        motion_time = ((phase * motion_len)/curr_fps).long() * curr_fps

        return motion_time

    def get_motion_length(self, motion_ids = None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def get_motion_num_frames(self, motion_ids = None):
        if motion_ids is None:
            return self._motion_num_frames
        else:
            return self._motion_num_frames[motion_ids]

    def get_motion_state(self, motion_ids, motion_times, offset = None):
        n = len(motion_ids)


        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        dof_pos0 = self.lrs[f0l]
        dof_pos1 = self.lrs[f1l]
        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        rg_pos0 = self.gts[f0l]
        rg_pos1 = self.gts[f1l]
        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]

        rg_vel0 = self.grvs[f0l]
        rg_vel1 = self.grvs[f1l]
        rg_ang_vel0 = self.gravs[f0l]
        rg_ang_vel1 = self.gravs[f1l]

        lp_pos0 = self.lps[f0l]
        lp_pos1 = self.lps[f1l]

        vals = [
             dof_pos0, dof_pos1,  rg_pos0, rg_pos1, dof_vel0, dof_vel1
        ]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)

        rg_pos = (1.0 - blend) * rg_pos0 + blend * rg_pos1 
        dof_pos = (1.0 - blend) * dof_pos0 + blend * dof_pos1
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
        left_ankle_pos = (1.0 - blend) * self.lal[f0l] + blend * self.lal[f1l]
        right_ankle_pos = (1.0 - blend) * self.ral[f0l] + blend * self.ral[f1l]
        key_pos = torch.stack([left_ankle_pos, right_ankle_pos], dim=1)
        rb_rot0 = torch_utils.exp_map_to_quat(rb_rot0)
        rb_rot1 = torch_utils.exp_map_to_quat(rb_rot1)
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend)
        rg_vel = (1.0 - blend) * rg_vel0 + blend * rg_vel1
        rg_ang_vel = (1.0 - blend) * rg_ang_vel0 + blend * rg_ang_vel1
        lp = (1.0 - blend_exp) * lp_pos0 + blend_exp * lp_pos1

        # self.torch_humanoid.fk_batch()

        return {
            "root_pos": rg_pos.clone(),
            "root_rot": rb_rot.clone(),
            "root_vel": rg_vel.clone(),
            "root_ang_vel": rg_ang_vel.clone(),
            "dof_pos": dof_pos.clone(),
            "dof_vel": dof_vel.clone(),
            "key_pos": key_pos,
            "local_pos": lp.clone(),
        }


    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0) # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        return self.num_bodies

    def _local_rotation_to_dof_smpl(self, local_rot):
        B, J, _ = local_rot.shape
        dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)

    # jp hack
    def _hack_test_vel_consistency(self, motion):
        test_vel = np.loadtxt("output/vel.txt", delimiter=",")
        test_root_vel = test_vel[:, :3]
        test_root_ang_vel = test_vel[:, 3:6]
        test_dof_vel = test_vel[:, 6:]

        dof_vel = motion.dof_vels
        dof_vel_err = test_dof_vel[:-1] - dof_vel[:-1]
        dof_vel_err = np.max(np.abs(dof_vel_err))

        root_vel = motion.global_root_velocity.numpy()
        root_vel_err = test_root_vel[:-1] - root_vel[:-1]
        root_vel_err = np.max(np.abs(root_vel_err))

        root_ang_vel = motion.global_root_angular_velocity.numpy()
        root_ang_vel_err = test_root_ang_vel[:-1] - root_ang_vel[:-1]
        root_ang_vel_err = np.max(np.abs(root_ang_vel_err))

        return
