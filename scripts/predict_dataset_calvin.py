import os
import h5py
import argparse
import yaml
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from icrt.models.policy.icrt_wrapper import ICRTWrapper
from icrt.util.eval_utils import EvalLogger

def load_hdf5(path):
    assert os.path.exists(path), f"Path {path} does not exist"
    def recursively_load_hdf5_group(group):
        data = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                data[key] = recursively_load_hdf5_group(item)
            else:
                data[key] = item[()]
        return data

    with h5py.File(path, 'r') as f:
        data = recursively_load_hdf5_group(f)
    return data

def get_data_from_h5(data, episode_name, resolution=(224, 224, 3), return_PIL_images=False):
    # img, action and proprio keys can be found in config/dataset_config_template.yaml
#     side_images = data[f"{episode_name}/observation/rgb_static"][:]
#     wrist_images = data[f"{episode_name}/observation/rgb_gripper"][:]
    # data is a dictionary with keys as episode names, observation, rgb_static.
    side_images = data[f"{episode_name}"]["observation"]["rgb_static"]
    wrist_images = data[f"{episode_name}"]["observation"]["rgb_gripper"]


    # action_keys = ["action/cartesian_pose", "action/gripper_position"]
    # proprio_keys = ["observation/cartesian_pose", "observation/gripper_position"]
    # actions = np.concatenate([data[f"{episode_name}/{key}"][:] for key in action_keys], axis=-1)
    # proprios = np.concatenate([data[f"{episode_name}/{key}"][:] for key in proprio_keys], axis=-1)
    action_keys = ["cartesian_pose", "gripper_position"]
    proprio_keys = ["cartesian_pose", "gripper_position"]
    actions = np.concatenate([data[f"{episode_name}"]['action'][key] for key in action_keys], axis=-1)
    proprios = np.concatenate([data[f"{episode_name}"]['observation'][key] for key in proprio_keys], axis=-1)

    side_images_l, wrist_images_l = [], []
    selected_idx = []
    for idx, (si, wi) in enumerate(zip(side_images, wrist_images)):
        if len(si) == 0 or len(wi) == 0:
            continue
        side_images_l.append(np.frombuffer(si, dtype="uint8").reshape(*si.shape))
        wrist_images_l.append(np.frombuffer(wi, dtype="uint8").reshape(*wi.shape))
        selected_idx.append(idx)

    # update the actions and proprios
    actions = actions[selected_idx]
    proprios = proprios[selected_idx]

    side_images_l = np.array(side_images_l)
    wrist_images_l = np.array(wrist_images_l)

    if return_PIL_images:
        side_images_l = [Image.fromarray(side_images_l[i]) for i in range(side_images_l.shape[0])]
        wrist_images_l = [Image.fromarray(wrist_images_l[i]) for i in range(wrist_images_l.shape[0])]

    return {
        "side_images": side_images_l,
        "wrist_images": wrist_images_l,
        "actions": actions,
        "proprios": proprios
    }

def get_prompt(data, task_name, n_prompt, indices=None):
    if indices is not None:
        assert len(indices) == n_prompt, f"Length of indices ({len(indices)}) does not match n_prompt ({n_prompt})"
    else:
        indices = [i for i in range(n_prompt)]
    prompt_data = []
    for index in indices:
        episode_name = f"task_{task_name}_episode_{index}"
        prompt = get_data_from_h5(data, episode_name, return_PIL_images=True)
        prompt_data.append(prompt)
    # concate the prompt data
    prompt_side_images = np.concatenate([prompt["side_images"] for prompt in prompt_data], axis=0)
    prompt_wrist_images = np.concatenate([prompt["wrist_images"] for prompt in prompt_data], axis=0)
    prompt_proprios = np.concatenate([prompt["proprios"] for prompt in prompt_data], axis=0)
    prompt_actions = np.concatenate([prompt["actions"] for prompt in prompt_data], axis=0)
    assert len(prompt_side_images) == len(prompt_wrist_images) == len(prompt_proprios) == len(prompt_actions), f"Lengths of prompt data do not match: {len(prompt_side_images)}, {len(prompt_wrist_images)}, {len(prompt_proprios)}, {len(prompt_actions)}"
    assert len(prompt_side_images.shape) == 4, f"prompt_side_images shape: {prompt_side_images.shape}"
    return prompt_side_images, prompt_wrist_images, prompt_proprios, prompt_actions

def main(args):
    # Load the config file
    with open(args.cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    dataset_paths = cfg['dataset_path']
    task_name = args.task_name
    dataset_paths = [path for path in dataset_paths if task_name in path]

    checkpoint_path = args.ckpt_path
    train_yaml_path = args.train_yaml_path
    vision_encoder_path = args.vision_encoder_path # if args.vision_encoder_path is not None else '/home/rutavms/research/gaze/icrt/checkpoints/crossmae_rtx/cross-mae-rtx-vitb.pth'
    icrt = ICRTWrapper(train_yaml_path, checkpoint_path, vision_encoder_path)
    resolution = (224, 224, 3)

    # Load the dataset
    data = load_hdf5(dataset_paths[0]) # only 0th dataset for now
    prompt_side_images, prompt_wrist_images, prompt_proprios, prompt_actions = get_prompt(data, task_name, args.n_prompt)
    if True:
        total_loss = 0
        for index in range(1, len(data.keys())):
            icrt.reset()
            icrt.prompt(
                prompt_side_images,
                prompt_wrist_images,
                prompt_proprios,
                prompt_actions,
            )
            episode_name = f"task_{task_name}_episode_{index}"
            eval_data = get_data_from_h5(data, episode_name, return_PIL_images=True)
            side_images, wrist_images, proprios, actions = eval_data["side_images"], eval_data["wrist_images"], eval_data["proprios"], eval_data["actions"]
            pred_actions = []
            loss_avg = 0
            l1_loss = nn.L1Loss()
            for i in range(len(side_images)):
                if i == 0:
                    action = None
                else:
                    action = actions[i-1:i]

                action = icrt(
                    side_images[i], wrist_images[i],
                    proprios[i:i+1],
                    action=action,
                    use_temporal=False,
                    teacher_forcing=True
                )
                pred_actions.append(action)

                loss = l1_loss(torch.from_numpy(action.reshape(1,-1)), torch.from_numpy(actions[i].reshape(1,-1)))
                if loss > 0.1:
                    print(f"Loss: {loss.item():.4f}, index: {i}")
                loss_avg += loss.item()
                # print(f"L1 loss: {loss.item():.4f}")

            loss_avg /= len(side_images)
            total_loss += loss_avg
            print(f"Average L1 loss: {loss_avg:.4f}")
        print(f"Total Average L1 loss: {total_loss / len(data.keys()):.4f}")
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--train_yaml_path", type=str, required=True)
    parser.add_argument("--cfg_path", type=str, required=True, help="path to the config file with the dataset paths")
    parser.add_argument("--vision_encoder_path", type=str, default=None)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--n_prompt", type=int, default=1)
    args = parser.parse_args()

    main(args)
