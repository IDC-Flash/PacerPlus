from ast import Try
import torch
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as sRot
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from mujoco_py import load_model_from_path
import mujoco_py
from uhc.utils.config_utils.copycat_config import Config as CC_Config
from uhc.khrylib.utils import get_body_qposaddr
from uhc.smpllib.smpl_mujoco import SMPL_BONE_ORDER_NAMES as joint_names
from uhc.smpllib.smpl_robot import Robot
from uhc.smpllib.smpl_local_robot import Robot as LocalRobot
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion
import scipy.ndimage.filters as filters
from typing import List, Optional
from tqdm import tqdm
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState

robot_cfg = {
    "mesh": False,
    "model": "smpl",
    "upright_start": True,
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
}
print(robot_cfg)

smpl_local_robot = LocalRobot(
    robot_cfg,
    data_dir="data/smpl",
    masterfoot=False,
)

amass_data = joblib.load(
    "/hdd/zen/data/ActBound/AMASS/amass_copycat_take5_5.pkl")
# amass_data = joblib.load("/hdd/zen/data/ActBound/AMASS/singles/amass_copycat_take5_singles_run.pkl")

mujoco_joint_names = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
    'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
    'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
    'R_Elbow', 'R_Wrist', 'R_Hand'
]

amss_run_data = [
    '0-ACCAD_Female1Running_c3d_C25 -  side step right_poses',
    '0-ACCAD_Female1Running_c3d_C5 - walk to run_poses',
    '0-ACCAD_Female1Walking_c3d_B15 - walk turn around (same direction)_poses',
    '0-ACCAD_Male1Walking_c3d_Walk B15 - Walk turn around_poses',
    '0-ACCAD_Male1Walking_c3d_Walk B16 - Walk turn change_poses',
    '0-ACCAD_Male2Running_c3d_C17 - run change direction_poses',
    '0-ACCAD_Male2Running_c3d_C20 - run to pickup box_poses',
    '0-ACCAD_Male2Running_c3d_C24 - quick sidestep left_poses',
    '0-ACCAD_Male2Running_c3d_C3 - run_poses',
    '0-ACCAD_Male2Walking_c3d_B15 -  Walk turn around_poses',
    '0-ACCAD_Male2Walking_c3d_B17 -  Walk to hop to walk a_poses',
    '0-ACCAD_Male2Walking_c3d_B18 -  Walk to leap to walk t2_poses',
    '0-ACCAD_Male2Walking_c3d_B18 -  Walk to leap to walk_poses',
    '0-BioMotionLab_NTroje_rub001_0017_circle_walk_poses',
    '0-BioMotionLab_NTroje_rub020_0027_circle_walk_poses',
    '0-BioMotionLab_NTroje_rub027_0027_circle_walk_poses',
    '0-BioMotionLab_NTroje_rub076_0027_circle_walk_poses',
    '0-BioMotionLab_NTroje_rub077_0027_circle_walk_poses',
    '0-BioMotionLab_NTroje_rub104_0027_circle_walk_poses',
    '0-Eyes_Japan_Dataset_aita_walk-04-fast-aita_poses',
    '0-Eyes_Japan_Dataset_aita_walk-21-one leg-aita_poses',
    '0-Eyes_Japan_Dataset_frederic_walk-04-fast-frederic_poses',
    '0-Eyes_Japan_Dataset_hamada_walk-06-catwalk-hamada_poses',
    '0-Eyes_Japan_Dataset_kaiwa_walk-27-thinking-kaiwa_poses',
    '0-Eyes_Japan_Dataset_shiono_walk-09-handbag-shiono_poses',
    '0-HumanEva_S2_Jog_1_poses', '0-HumanEva_S2_Jog_3_poses',
    '0-KIT_10_WalkInClockwiseCircle10_poses',
    '0-KIT_10_WalkInCounterClockwiseCircle05_poses',
    '0-KIT_10_WalkInCounterClockwiseCircle10_poses',
    '0-KIT_12_WalkInClockwiseCircle09_poses',
    '0-KIT_12_WalkInClockwiseCircle11_poses',
    '0-KIT_12_WalkInCounterClockwiseCircle01_poses',
    '0-KIT_12_WalkingStraightForwards03_poses', '0-KIT_167_downstairs04_poses',
    '0-KIT_167_upstairs_downstairs01_poses',
    '0-KIT_167_walking_medium04_poses', '0-KIT_167_walking_run02_poses',
    '0-KIT_167_walking_run06_poses', '0-KIT_183_run04_poses',
    '0-KIT_183_upstairs10_poses', '0-KIT_183_walking_fast03_poses',
    '0-KIT_183_walking_fast05_poses', '0-KIT_183_walking_medium04_poses',
    '0-KIT_183_walking_run04_poses', '0-KIT_183_walking_run05_poses',
    '0-KIT_183_walking_run06_poses', '0-KIT_205_walking_medium04_poses',
    '0-KIT_205_walking_medium10_poses', '0-KIT_314_run02_poses',
    '0-KIT_314_run04_poses', '0-KIT_314_walking_fast06_poses',
    '0-KIT_314_walking_medium02_poses', '0-KIT_314_walking_medium07_poses',
    '0-KIT_314_walking_slow05_poses', '0-KIT_317_walking_medium09_poses',
    '0-KIT_348_walking_medium07_poses', '0-KIT_348_walking_run10_poses',
    '0-KIT_359_downstairs04_poses', '0-KIT_359_downstairs06_poses',
    '0-KIT_359_upstairs09_poses', '0-KIT_359_upstairs_downstairs03_poses',
    '0-KIT_359_walking_fast10_poses', '0-KIT_359_walking_run05_poses',
    '0-KIT_359_walking_slow02_poses', '0-KIT_359_walking_slow09_poses',
    '0-KIT_3_walk_6m_straight_line04_poses', '0-KIT_3_walking_medium07_poses',
    '0-KIT_3_walking_medium08_poses', '0-KIT_3_walking_run03_poses',
    '0-KIT_3_walking_slow08_poses', '0-KIT_424_run05_poses',
    '0-KIT_424_upstairs03_poses', '0-KIT_424_upstairs05_poses',
    '0-KIT_424_walking_fast04_poses', '0-KIT_425_walking_fast01_poses',
    '0-KIT_425_walking_fast04_poses', '0-KIT_425_walking_fast05_poses',
    '0-KIT_425_walking_medium08_poses',
    '0-KIT_4_WalkInClockwiseCircle02_poses',
    '0-KIT_4_WalkInClockwiseCircle05_poses',
    '0-KIT_4_WalkInCounterClockwiseCircle02_poses',
    '0-KIT_4_WalkInCounterClockwiseCircle07_poses',
    '0-KIT_4_WalkInCounterClockwiseCircle08_poses',
    '0-KIT_513_downstairs06_poses', '0-KIT_513_upstairs07_poses',
    '0-KIT_675_walk_with_handrail_table_left003_poses',
    '0-KIT_6_WalkInClockwiseCircle04_1_poses',
    '0-KIT_6_WalkInClockwiseCircle05_1_poses',
    '0-KIT_6_WalkInCounterClockwiseCircle01_1_poses',
    '0-KIT_6_WalkInCounterClockwiseCircle10_1_poses',
    '0-KIT_7_WalkInCounterClockwiseCircle09_poses',
    '0-KIT_7_WalkingStraightForwards04_poses',
    '0-KIT_8_WalkInCounterClockwiseCircle03_poses',
    '0-KIT_8_WalkInCounterClockwiseCircle05_poses',
    '0-KIT_8_WalkInCounterClockwiseCircle10_poses',
    '0-KIT_9_WalkInClockwiseCircle04_poses',
    '0-KIT_9_WalkInCounterClockwiseCircle05_poses',
    '0-KIT_9_WalkingStraightForwards01_poses',
    '0-KIT_9_WalkingStraightForwards04_poses', '0-KIT_9_run01_poses',
    '0-KIT_9_run05_poses', '0-KIT_9_walking_medium02_poses',
    '0-KIT_9_walking_run02_poses', '0-KIT_9_walking_slow07_poses',
    '0-SFU_0005_0005_Jogging001_poses', '0-TotalCapture_s4_walking2_poses',
    '0-Transitions_mocap_mazen_c3d_crouchwalk_running_poses'
]

amass_full_motion_dict = {}
for key_name in tqdm(amass_data.keys()):
    smpl_data_entry = amass_data[key_name]
    file_name = f"data/amass/singles/{key_name}.npy"
    B = smpl_data_entry['pose_aa'].shape[0]
    # if  not(not "treadmill" in key_name.lower() and not "normal" in key_name.lower() and not "back" in key_name.lower() and \
    #     ("walk" in key_name.lower() or "run" in key_name.lower() or "jog" in key_name.lower() or "stairs" in key_name.lower())):
    #     continue
    if not key_name in amss_run_data:
        continue
    # if not "0-ACCAD_Female1Running_c3d_C5 - walk to run_poses" in key_name:
    #     continue
    if B > 100:
        start, end = 45, 0
    else:
        start, end = 0, 0

    print("using data offset")
    pose_aa = smpl_data_entry['pose_aa'].copy()[start:]
    root_trans = smpl_data_entry['trans'].copy()[start:]
    B = pose_aa.shape[0]

    # pose_aa = smpl_data_entry['pose_aa'].copy()
    # root_trans = smpl_data_entry['trans'].copy()

    beta = smpl_data_entry['beta'].copy()
    gender = smpl_data_entry['gender']
    fps = 30.0

    if isinstance(gender, np.ndarray):
        gender = gender.item()

    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    if gender == "neutral":
        gender_number = [0]
    elif gender == "male":
        gender_number = [1]
    elif gender == "female":
        gender_number = [2]
    else:
        import ipdb
        ipdb.set_trace()
        raise Exception("Gender Not Supported!!")
    smpl_2_mujoco = [
        joint_names.index(q) for q in mujoco_joint_names if q in joint_names
    ]
    batch_size = pose_aa.shape[0]
    pose_aa = np.concatenate(
        [pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)
        
    pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(
        batch_size, 24, 4)[..., smpl_2_mujoco, :]

    gender_number, beta[:], gender = [0], 0, "neutral"; print("using neutral model")

    smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None, ]),
                                        gender=gender_number,
                                        objs_info=None)
    smpl_local_robot.write_xml("amp/data/assets/mjcf/smpl_humanoid_1.xml")
    skeleton_tree = SkeletonTree.from_mjcf(
        "amp/data/assets/mjcf/smpl_humanoid_1.xml")

    root_trans_offset = torch.from_numpy(
        root_trans) + skeleton_tree.local_translation[0]

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat),
        root_trans_offset,
        is_local=True)
    if robot_cfg['upright_start']:
        pose_quat_global = (
            sRot.from_quat(
                new_sk_state.global_rotation.reshape(-1, 4).numpy()) *
            sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(
                B, -1, 4)
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat_global),
            root_trans_offset,
            is_local=False)

    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)
    new_motion_out = new_motion.to_dict()
    new_motion_out['pose_aa'] = pose_aa
    new_motion_out['pose_quat'] = pose_quat
    new_motion_out['beta'] = beta
    new_motion_out['gender'] = gender
    new_motion_out['__name__'] = "SkeletonMotion"
    amass_full_motion_dict[key_name] = new_motion_out

import ipdb

ipdb.set_trace()
# joblib.dump(amass_full_motion_dict, "data/amass/pkls/amass_isaac_take1.pkl")
# joblib.dump(amass_full_motion_dict, "data/amass/pkls/amass_isaac_locomotion_upright.pkl")
# joblib.dump(amass_full_motion_dict, "data/amass/pkls/amass_isaac_locomotion_zero.pkl")
# joblib.dump(amass_full_motion_dict, "data/amass/pkls/amass_isaac_locomotion_zero_upright.pkl")
# joblib.dump(amass_full_motion_dict, "data/amass/pkls/amass_isaac_run.pkl")
# joblib.dump(amass_full_motion_dict,"data/amass/pkls/amass_isaac_run_upright.pkl")
joblib.dump(amass_full_motion_dict, "data/amass/pkls/amass_isaac_run_upright_zero.pkl")
# joblib.dump(amass_full_motion_dict,
# "data/amass/pkls/amass_isaac_slowalk_upright.pkl")
# joblib.dump(amass_full_motion_dict, "data/amass/pkls/amass_isaac_slowalk.pkl")
