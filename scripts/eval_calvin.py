import os
import cv2
import h5py
import torch
import copy
import argparse
import time
import torch.nn as nn
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from easydict import EasyDict
from tqdm import trange
from collections import Counter
from termcolor import colored

import hydra
from pathlib import Path
from omegaconf import OmegaConf

import calvin_agent
from calvin_env.envs.play_table_env import get_env
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

from icrt.util.calvin_utils import \
    generate_dataset_paths, \
    get_subsequence, break_subsequence, \
    get_conf_path, \
    taskname2taskname, get_initial_states
from icrt.models.policy.icrt_wrapper import ICRTWrapper
from icrt.util.eval_utils import EvalLogger

def get_task_to_id_dict(dataroot, task_name, annotations):
    dataset_paths = []
    return_list = []
    for _i, ((start, end), ann) in enumerate(zip(annotations['info']['indx'], annotations['language']['task'])):
        if ann != task_name:
            continue
        dataset_paths = generate_dataset_paths(dataroot, start, end)
        trajectory = get_subsequence(dataset_paths)
        actions, proprio, obs = break_subsequence(trajectory)
        init_state = get_initial_states(trajectory)
        return_list.append((obs, proprio, actions, init_state))
    return return_list

def get_env_datamodule(args):
    task_name = "task_D_D"
    mode = "training" if args.mode == 'train' else 'validation'
    # task_name = "calvin_debug_dataset"
    confg_path = get_conf_path()

    datamodule_default = OmegaConf.load('./config/calvin/datamodule.yaml')
    datamodule_default.root_data_dir = os.path.join(os.environ["CALVIN_DATAROOT"], task_name)
    datamodule_default.datasets.vision_dataset.datasets_dir = os.path.join(os.environ["CALVIN_DATAROOT"], task_name, mode)
    datamodule_default.datasets.lang_dataset.datasets_dir = os.path.join(os.environ["CALVIN_DATAROOT"], task_name, mode)
    datamodule_default = OmegaConf.create(datamodule_default)

    print(OmegaConf.to_yaml(datamodule_default))

    assert datamodule_default.datasets is not None
    assert datamodule_default.proprioception_dims is not None
    data_module = hydra.utils.instantiate(datamodule_default, _recursive_=False)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["lang"]
    print("done preparing data")

    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    # base path is calvin/calvin_models/conf
    rollout_cfg = OmegaConf.load(os.path.join(get_conf_path(), "callbacks/rollout/default.yaml"))
    env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)
    return env, data_module, os.path.join(os.environ["CALVIN_DATAROOT"], task_name, mode)

def visualize_trajectory(obs):
    # for d in obs:
    index = 0
    while True:
        cv2.imshow("image", obs["rgb_static"][index][:, :, ::-1])
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        elif key == 83:  # Right arrow
            index = (index + 1) % len(obs["rgb_static"])
        elif key == 81:  # Left arrow
            index = (len(obs["rgb_static"]) + index - 1) % len(obs["rgb_static"])
        else:
            print(f'Unrecognized keycode "{key}"')
        if index % len(obs["rgb_static"]) == 0:
            print("End of dataset reached")
            break
    cv2.destroyAllWindows()
    return

def preprocess_obs(obs):
    # convert everything to numpy arrays
    for key in obs:
        if isinstance(obs[key], torch.Tensor):
            obs[key] = obs[key].cpu().numpy().squeeze()
        if isinstance(obs[key], dict):
            for subkey in obs[key]:
                if isinstance(obs[key][subkey], torch.Tensor):
                    obs[key][subkey] = obs[key][subkey].cpu().numpy().squeeze()
    return obs

def rollout(icrt, env, initial_state, task_oracle, task, args, gt_actions=None, gt_obs=None, gt_proprio=None, use_gt=False):
    # state_obs, rgb_obs, depth_obs = episode["robot_obs"], episode["rgb_obs"], episode["depth_obs"]
    obs = env.reset(**initial_state)
    # get lang annotation for subtask
    # lang_annotation = val_annotations[task][0]

    # model.reset()
    start_info = env.get_info()
    loss =  nn.L1Loss()
    avg_loss = 0.0

    for step in range(args.ep_len):
        # action = model.step(obs, lang_annotation)
        action = None

        ##### ICRT PREPARATION #####
        '''
        obs = preprocess_obs(obs)
        side_image = Image.fromarray(obs['rgb_obs']["rgb_static"].astype(np.uint8).transpose(1, 2, 0))
        wrist_image = Image.fromarray(obs["rgb_obs"]["rgb_gripper"].astype(np.uint8).transpose(1, 2, 0))
        assert len(obs["robot_obs"].shape) == 1
        proprio = np.concatenate([obs["robot_obs"][:6], obs["robot_obs"][-1:]], axis=0)
        pred_action = icrt(
            side_image, wrist_image,
            proprio.reshape(1, -1),
            action=None,
            use_temporal=False,
            teacher_forcing=False,
        )
        '''
        side_image = Image.fromarray(gt_obs["rgb_static"][step].astype(np.uint8))
        wrist_image = Image.fromarray(gt_obs["rgb_gripper"][step].astype(np.uint8))
        proprio = np.concatenate([gt_proprio[0][step][:6], gt_proprio[1][step][-1:]], axis=0)
        pred_action = icrt(
            side_image, wrist_image,
            proprio.reshape(1, -1),
            action=None,
            use_temporal=False,
            teacher_forcing=False,
        )


        if use_gt:
            action = gt_actions[step]
        else:
            action = pred_action

        loss_val = 0.0
        if step < len(gt_actions):
            loss_val = loss(torch.tensor(pred_action), torch.tensor(gt_actions[step])).item()
        else:
            loss_val = loss(torch.tensor(pred_action), torch.tensor(gt_actions[-1])).item()
        avg_loss += loss_val
        # print(loss_val)

        obs, _, _, current_info = env.step(action)
        if args.debug:
            img = env.render()
            time.sleep(0.1)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task})
        if len(current_task_info) > 0:
            if args.debug:
                print(colored("S", "green"), end=" ")
            return True, avg_loss / (step + 1)

        if use_gt:
            if step == len(gt_actions) - 1:
                if args.debug:
                    print(colored("F", "red"), end=" ")
                return False, avg_loss / (step + 1)
    if args.debug:
        print(colored("F", "red"), end=" ")
    return False, avg_loss / args.ep_len

def generate_prompt(traj_list, index=0):
    traj = traj_list[index] # get the first trajectory
    obs, proprio, actions, init_state = traj
    assert isinstance(proprio, tuple)
    assert isinstance(actions, tuple)
    actions = np.concatenate([actions[0], actions[1]], axis=1)
    proprio = np.concatenate([proprio[0], proprio[1]], axis=1)
    # return PIL images
    prompt_side_images = [Image.fromarray(o) for o in obs["rgb_static"]]
    prompt_gripper_images = [Image.fromarray(o) for o in obs["rgb_gripper"]]
    return {
        "side_images": prompt_side_images,
        "wrist_images": prompt_gripper_images,
        "proprios": proprio,
        "actions": actions,
    }

def evaluate_policy_singlestep(env, icrt, datamodule, dataroot, args):
    # conf_dir = Path(__file__).absolute().parents[2] / "conf"
    conf_dir = get_conf_path()
    task_cfg = OmegaConf.load(os.path.join(conf_dir, "callbacks/rollout/tasks/new_playtable_tasks.yaml"))
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(os.path.join(conf_dir, "annotations/new_playtable_validation.yaml"))
    lang_annotations = np.load(os.path.join(dataroot, "lang_annotations", "auto_lang_ann.npy"), allow_pickle=True).item()

    # task_to_id_dict = torch.load(checkpoint)["task_to_id_dict"]
    # dataset = datamodule.val_dataloader().dataset.datasets["vis"]

    results = Counter()

    total_trajs = 0
    task_to_id_dict = task_oracle.task_to_id
    global_avg_loss = 0.0

    for task_name, task_id in task_to_id_dict.items():
        if task_name == "rotate_red_block_left":
            continue
        task_id = task_to_id_dict[task_name]
        traj_list = get_task_to_id_dict(dataroot, task_name, lang_annotations)
        if len(traj_list) == 0:
            continue
        print(f"Task: {task_name}")
        # visualize_trajectory(traj_list[0][0])
        prompt_data = [generate_prompt(traj_list, index=i) for i in range(args.n_prompt)]
        prompt_side_images = np.concatenate([np.array(p["side_images"]) for p in prompt_data], axis=0)
        prompt_wrist_images = np.concatenate([np.array(p["wrist_images"]) for p in prompt_data], axis=0)
        prompt_proprios = np.concatenate([p["proprios"] for p in prompt_data], axis=0)
        prompt_actions = np.concatenate([p["actions"] for p in prompt_data], axis=0)
        prompt_dict = {
            "side_image": prompt_side_images,
            "wrist_image": prompt_wrist_images,
            "proprio": prompt_proprios,
            "action": prompt_actions,
        }
        for traj in traj_list:
            print(f"Task: {task_name}")
            total_trajs += 1
            obs, proprio, actions, init_state = traj
            actions = np.concatenate([actions[0], actions[1]], axis=1)
            icrt.reset()
            icrt.prompt(**prompt_dict)
            success, avg_loss = rollout(icrt, env, init_state, task_oracle=task_oracle, task=task_name, args=args, gt_actions=actions, gt_obs=obs, gt_proprio=proprio, use_gt=args.use_gt)
            global_avg_loss += avg_loss
            print(colored(f"Success: {success}, Avg Loss: {avg_loss:.3f}", "yellow"))
            if success:
                results[task_name] += 1
            # visualize_trajectory(obs)
    print(f"Total eval trajs: {total_trajs}")
    print(f"Global Avg Loss: {global_avg_loss / total_trajs:.3f}")
    print(f"SR: {sum(results.values()) / total_trajs * 100:.1f}%")

def main(args):
    datamodule = None
    data_name = "task_D_D"
    dataroot = os.path.join(os.environ["CALVIN_DATAROOT"], data_name, 'training' if args.mode=='train' else 'validation')
    # env = get_env(dataroot, show_gui=False)
    env, datamodule, dataroot = get_env_datamodule(args)
    env.reset()

    checkpoint_path = args.ckpt_path
    train_yaml_path = args.train_yaml_path
    vision_encoder_path = args.vision_encoder_path # if args.vision_encoder_path is not None else '/home/rutavms/research/gaze/icrt/checkpoints/crossmae_rtx/cross-mae-rtx-vitb.pth'
    if vision_encoder_path is not None:
        print(colored("Using vision encoder", "green"))
        print(args.vision_encoder_path)
    icrt = ICRTWrapper(train_yaml_path, checkpoint_path, vision_encoder_path)
    resolution = (224, 224, 3)
    icrt.reset() # do not cache history for this example

    evaluate_policy_singlestep(env, icrt, datamodule, dataroot, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--train_yaml_path", type=str, required=True)
    parser.add_argument("--vision_encoder_path", type=str, default=None)
    parser.add_argument("--n_eval", type=int, default=10)
    parser.add_argument("--mode", type=str, default="val")
    parser.add_argument("--n_prompt", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_gt", action="store_true")
    args = parser.parse_args()
    args.ep_len = 120

    main(args)
