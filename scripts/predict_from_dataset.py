import os
import h5py
import torch
import argparse
import torch.nn as nn
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from easydict import EasyDict
from tqdm import trange

import robosuite.utils.transform_utils as T
import robomimic.utils.obs_utils as ObsUtils

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.algos import *
from libero.lifelong.datasets import get_dataset, SequenceVLDataset, GroupedTaskDataset
from libero.lifelong.metric import (
    evaluate_loss,
    evaluate_success,
)
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    NpEncoder,
    compute_flops,
)
from icrt.models.policy.icrt_wrapper import ICRTWrapper

benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
    "libero_single": "LIBERO_SINGLE",
}

class EvalLogger:
    def __init__(self, log_keys: list):
        self.log_keys = log_keys
        self._dict = {}
        for key in self.log_keys:
            self._dict.update({key: []})

    def add_kv(self, key, value):
        assert key in self.log_keys, f"Tried to log {key} but logger is initialized with keys {self.log_keys}"
        self._dict[key].append(value)

    def save(self, filename):
        df = pd.DataFrame(self._dict)
        # if the file exists, append to it
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)
        return

def get_task_from_name(task_name, benchmark):
    task = None
    for task_id in range(benchmark.n_tasks):
        task = benchmark.get_task(task_id)
        print(f"Task {task_id}: {task.language}")
        if task.name == task_name:
            break
        elif task_id == benchmark.n_tasks - 1:
            raise ValueError(f"Task {task_name} not found in benchmark {benchmark_name}")
    return task, task_id

def raw_obs_to_tensor_obs(obs, cfg):
    """
    Prepare the tensor observations as input for the algorithm.
    """
    env_num = len(obs)

    data = {
        "observation": {},
    }

    all_obs_keys = []
    for modality_name, modality_list in cfg.data.obs.modality.items():
        for obs_name in modality_list:
            data["observation"][obs_name] = []
        all_obs_keys += modality_list

    for k in range(env_num):
        for obs_name in all_obs_keys:
            data["observation"][obs_name].append(
                # ObsUtils.process_obs(
                    torch.from_numpy(obs[k][cfg.data.obs_key_mapping[obs_name]]),
                    # obs_key=obs_name,
                # ).float()
            )

    # set gripper_position as the first element of robot0_gripper_qpos
    data["observation"]["gripper_position"] = [torch.from_numpy(obs[k]["robot0_gripper_qpos"][:1]) for k in range(env_num)]
    # calculate cartesion_pose from eef_pos and eef_quat
    data["observation"]["cartesian_pose"] = [
        torch.from_numpy(np.concatenate(
            (obs[k]["robot0_eef_pos"], T.mat2euler(T.quat2mat(obs[k]["robot0_eef_quat"]))), axis=-1
        )) for k in range(env_num)
    ]

    for key in data["observation"]:
        data["observation"][key] = torch.stack(data["observation"][key])

    # data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))
    return data

def preprocess_libero_obs(obs, cfg, env):
    data_obs = raw_obs_to_tensor_obs(obs, cfg)
    # convert the naming convention to observation/key
    # remove the batch dimension and make it numpy
    data_obs = {f"observation/{k}": v[0].numpy() for k, v in data_obs["observation"].items()}
    return data_obs

def set_cfg_libero(libero_folder, libero_init_states_folder):
    cfg = {}
    cfg = EasyDict(cfg)
    cfg.folder = libero_folder
    cfg.data = EasyDict(); cfg.data.obs = EasyDict(); cfg.data.obs.modality = EasyDict()
    cfg.data.obs.modality.rgb = ["agentview_rgb", "eye_in_hand_rgb"]
    cfg.data.obs.modality.depth = []
    cfg.data.obs.modality.low_dim = []
    cfg.data.obs_key_mapping = EasyDict()
    cfg.data.obs_key_mapping["agentview_rgb"] = "agentview_image"
    cfg.data.obs_key_mapping["eye_in_hand_rgb"] = "robot0_eye_in_hand_image"
    cfg.init_states_folder = libero_init_states_folder
    cfg.data.seq_len = 10 # doesn't matter
    return cfg

def get_data_from_h5(data, episode_name, resolution=(224, 224, 3), return_PIL_images=False):
    # img, action and proprio keys can be found in config/dataset_config_template.yaml
    side_images = data[f"{episode_name}/observation/agentview_rgb"][:]
    wrist_images = data[f"{episode_name}/observation/eye_in_hand_rgb"][:]

    action_keys = ["action/cartesian_pose", "action/gripper_position"]
    proprio_keys = ["observation/cartesian_pose", "observation/gripper_position"]
    actions = np.concatenate([data[f"{episode_name}/{key}"][:] for key in action_keys], axis=-1)
    proprios = np.concatenate([data[f"{episode_name}/{key}"][:] for key in proprio_keys], axis=-1)

    side_images_l, wrist_images_l = [], []
    selected_idx = []
    for idx, (si, wi) in enumerate(zip(side_images, wrist_images)):
        if len(si) == 0 or len(wi) == 0:
            continue
        side_images_l.append(np.frombuffer(si, dtype="uint8").reshape(resolution))
        wrist_images_l.append(np.frombuffer(wi, dtype="uint8").reshape(resolution))
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
def main(args):
    checkpoint_path = args.ckpt_path
    train_yaml_path = args.train_yaml_path
    vision_encoder_path = args.vision_encoder_path if args.vision_encoder_path is not None else '/home/rutavms/research/gaze/icrt/checkpoints/crossmae_rtx/cross-mae-rtx-vitb.pth'
    icrt = ICRTWrapper(train_yaml_path, checkpoint_path, vision_encoder_path)

    task_name_prompt = args.prompt_task_name if args.prompt_task_name is not None else args.task_name
    # task_name_prompt = "open_the_middle_drawer_of_the_cabinet"
    dataset_path = f"/home/rutavms/data/icrt/libero_goal/{task_name_prompt}_demo_icrt.hdf5"
    data = h5py.File(dataset_path, "r")
    resolution = (224, 224, 3)
    episode_name = f"task_{task_name_prompt}_demo_49"
    print("selected episode: ", episode_name)
    obs_dict = get_data_from_h5(data, episode_name, resolution, return_PIL_images=True)
    prompt_side_images, prompt_wrist_images, prompt_proprios, prompt_actions = obs_dict["side_images"], obs_dict["wrist_images"], obs_dict["proprios"], obs_dict["actions"]

    icrt.reset() # do not cache history for this example
    icrt.prompt(
        prompt_side_images,
        prompt_wrist_images,
        prompt_proprios,
        prompt_actions,
    )

    task_name = args.task_name
    dataset_path = f"/home/rutavms/data/icrt/libero_goal/{task_name}_demo_icrt.hdf5"
    if False:
        data = h5py.File(dataset_path, "r")
        episode_name = f"task_{task_name}_demo_1"
        print("selected episode: ", episode_name)

        obs_dict = get_data_from_h5(data, episode_name, resolution, return_PIL_images=True)
        side_images, wrist_images, proprios, actions = obs_dict["side_images"], obs_dict["wrist_images"], obs_dict["proprios"], obs_dict["actions"]

        pred_actions = []
        loss_avg = 0
        l1_loss = nn.L1Loss()
        for i in trange(len(side_images)):
            if i == 0:
                action = None
            else:
                action = actions[i-1:i]
            # When rolling out on the robot, make sure to set teacher_forcing=False and action=None, since we don't have the ground truth action, and the previously generated action is cached by ICRTWrapper.
            # side_images, wrist_images are PIL images
            # proprios: 1, 7
            # action: 1,
            action = icrt(
                side_images[i], wrist_images[i],
                proprios[i:i+1],
                action=action,
                use_temporal=False,
                teacher_forcing=True
            )
            pred_actions.append(action)

            # import ipdb; ipdb.set_trace()
            loss = l1_loss(torch.from_numpy(action.reshape(1,-1)), torch.from_numpy(actions[i].reshape(1,-1)))
            loss_avg += loss.item()
            # print(f"L1 loss: {loss.item():.4f}")

        loss_avg /= len(side_images)
        print(f"Average L1 loss: {loss_avg:.4f}")


    benchmark_name = 'libero_goal'
    libero_folder = get_libero_path("datasets")
    libero_bddl_folder = get_libero_path("bddl_files")
    libero_init_states_folder = get_libero_path("init_states")
    benchmark = get_benchmark(benchmark_name)(0)

    task, task_id = get_task_from_name(task_name, benchmark)
    cfg = set_cfg_libero(libero_folder, libero_init_states_folder)
    _, shape_meta = get_dataset(
        dataset_path=os.path.join(
            cfg.folder, benchmark.get_task_demonstration(task_id)
        ),
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
    )
    print("")
    # get the parent folder of the checkpoint path and add videos
    video_folder = os.path.join(os.path.dirname(checkpoint_path), f"videos_{task_name}"); os.makedirs(video_folder, exist_ok=True)
    log_folder = os.path.join(os.path.dirname(checkpoint_path), f"logs"); os.makedirs(log_folder, exist_ok=True)
    keys_to_log = ["task_name", "prompt_task_name", "task_id", "n_eval", "ckpt", "success_rate"]
    eval_logger = EvalLogger(keys_to_log)
    log_filename = os.path.join(log_folder, f"eval_logs.csv")

    with Timer() as t, VideoWriter(video_folder, True) as video_writer:
        env_args = {
            "bddl_file_name": os.path.join(
                libero_bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": 224,
            "camera_widths": 224,
        }
        env_num = 1
        assert env_num == 1
        env = SubprocVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
        )
        env.reset()
        env.seed(0)
        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        num_episodes = args.n_eval
        num_success = 0
        for ep_idx in range(num_episodes//env_num):
            steps = 0
            dones = [False] * env_num
            indices = np.arange(ep_idx*env_num, ep_idx*env_num + env_num, 1) % init_states.shape[0]
            print("indices: ", indices)
            init_states_ = init_states[indices]
            env.reset()
            obs = env.set_init_state(init_states_)
            for _ in range(5):  # simulate the physics without any actions
                env.step(np.zeros((env_num, 7)))
            assert env_num == 1, "env_num must be 1 for now"

            with torch.no_grad():
                icrt.reset()
                icrt.prompt(
                    prompt_side_images,
                    prompt_wrist_images,
                    prompt_proprios,
                    prompt_actions,
                )
                pred_actions = []
                while steps < 600:
                    steps += 1
                    obs = preprocess_libero_obs(obs, cfg, env)
                    # import ipdb; ipdb.set_trace()
                    # side_image = Image.fromarray(np.frombuffer(obs["observation/agentview_rgb"], dtype="uint8").reshape(224, 224, 3))
                    side_image = Image.fromarray(obs["observation/agentview_rgb"].reshape(224, 224, 3))
                    wrist_image = Image.fromarray(obs["observation/eye_in_hand_rgb"].reshape(224, 224, 3))
                    proprio = np.concatenate([obs["observation/cartesian_pose"], obs["observation/gripper_position"]], axis=-1)
                    if steps == 1:
                        action = None
                    else:
                        action = pred_actions[steps-2:steps-1]

                    # import ipdb; ipdb.set_trace()
                    action = icrt(
                        side_image, wrist_image,
                        proprio.reshape(1, -1),
                        action=None,
                        use_temporal=False,
                        teacher_forcing=False,
                    )

                    action = action.reshape(1, -1)
                    # import ipdb; ipdb.set_trace()
                    obs, reward, done, info = env.step(action)
                    video_writer.append_vector_obs(
                        obs, dones, camera_name="agentview_image"
                    )
                    pred_actions.append(action)
                    success = env.check_success()
                    # print("Success: ", success)
                    # check whether succeed
                    for k in range(env_num):
                        dones[k] = dones[k] or done[k]
                    if all(dones):
                        break
                for k in range(env_num):
                    num_success += int(dones[k])

    eval_logger.add_kv("task_name", task_name)
    eval_logger.add_kv("prompt_task_name", task_name_prompt)
    eval_logger.add_kv("ckpt", os.path.basename(checkpoint_path))
    eval_logger.add_kv("task_id", task_id)
    eval_logger.add_kv("n_eval", num_episodes)
    eval_logger.add_kv("success_rate", num_success / num_episodes)
    eval_logger.save(log_filename)
    print(f"Success rate: {num_success / num_episodes:.4f}")
    success_rate = num_success / num_episodes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--train_yaml_path", type=str, required=True)
    parser.add_argument("--vision_encoder_path", type=str, default=None)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--prompt_task_name", type=str, required=False, default=None)
    parser.add_argument("--n_eval", type=int, default=10)
    args = parser.parse_args()

    main(args)
