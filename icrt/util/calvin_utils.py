# Description: Utility functions for Calvin dataset
import os
import json
import numpy as np
import calvin_agent

def get_conf_path():
    return os.path.join(calvin_agent.__path__[0], "../conf")

def generate_dataset_paths(dataroot, start, end):
    dataset_paths = []
    for i in range(start, end):
        episode_path = os.path.join(dataroot, f"episode_{i:07d}.npz")
        dataset_paths.append(episode_path)
    return dataset_paths

def get_subsequence(state_paths):
    # find the dataset index
    subsequence = []
    for state_path in state_paths:
        with np.load(state_path) as f:
            data = {k: f[k] for k in f.files}
        subsequence.append(data)
    return subsequence

def load_proprio(subseq):
    # returns the cartesian_pose and gripper_position
    cartesian_pose = [s['robot_obs'][None, :6] for s in subseq] # (T, 6)
    cartesian_pose = np.concatenate(cartesian_pose, axis=0) # (T, 6)

    gripper_position = [s['robot_obs'][None, -1:] for s in subseq] # (T, 1)
    gripper_position = np.concatenate(gripper_position, axis=0) # (T, 1)
    return cartesian_pose, gripper_position

def load_action(subseq):
    # returns the action/cartesian_pose, action/gripper_position
    cartesian_pose = [s['actions'][None, :6] for s in subseq] # (T, 6)
    cartesian_pose = np.concatenate(cartesian_pose, axis=0) # (T, 6)

    gripper_position = [s['actions'][None, -1:] for s in subseq] # (T, 1)
    gripper_position = np.concatenate(gripper_position, axis=0) # (T, 1)
    return cartesian_pose, gripper_position

def load_obs(subseq, keys=["rgb_gripper", "rgb_static"]):
    key2obs_img = {}
    for key in keys:
        obs_img = [s[key][None] for s in subseq]
        obs_img = np.concatenate(obs_img, axis=0)
        key2obs_img[key] = obs_img
    return key2obs_img

def break_subsequence(subsequence):
    actions = load_action(subsequence)
    proprio = load_proprio(subsequence)
    obs = load_obs(subsequence)
    return actions, proprio, obs

def get_initial_states(subsequence):
    init_state_dict = {'robot_obs': subsequence[0]['robot_obs'], 'scene_obs': subsequence[0]['scene_obs']}
    return init_state_dict

def taskname2taskname(task_ann):
    possible_color_names = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'magenta', 'white', 'black', 'pink']
    for color in possible_color_names:
        task_ann = task_ann.replace(color + '_', "")
    return task_ann
