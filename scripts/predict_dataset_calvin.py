import os
import time
import h5py
import argparse
import yaml
import numpy as np
from PIL import Image
from easydict import EasyDict
from tqdm import trange
from collections import Counter
from termcolor import colored
from collections import defaultdict
from pathlib import Path

import timm
import hydra
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf

from pytorch_lightning import seed_everything

import calvin_agent
from calvin_env.envs.play_table_env import get_env
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)

from icrt.util.calvin_utils import (
    generate_dataset_paths,
    get_subsequence, break_subsequence,
    get_conf_path,
    taskname2taskname,
    get_initial_states
)
from icrt.data import load_datasets
from icrt.models.policy.icrt_wrapper import ICRTWrapper
from icrt.util.eval_utils import EvalLogger, visualize_trajectory

def make_env(dataset_path, obs_space=None, show_gui=False):
    val_folder = Path(dataset_path)
    env = get_env(val_folder, obs_space=obs_space, show_gui=show_gui)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env

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

def get_task_to_id_dict(dataroot, task_name, annotations):
    dataset_paths = []
    return_list = []
    for _i, ((start, end), ann) in enumerate(zip(annotations['info']['indx'], annotations['language']['task'])):
        if ann != task_name:
            continue
        print(f"Episode {_i}: {ann}")
        dataset_paths = generate_dataset_paths(dataroot, start, end)
        trajectory = get_subsequence(dataset_paths)
        actions, proprio, obs = break_subsequence(trajectory)
        init_state = get_initial_states(trajectory)
        return_list.append((obs, proprio, actions, init_state))
    return return_list

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

def load_vision_transforms(args):
    pretrained_cfg = {'url': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth', 'hf_hub_id': 'timm/vit_base_patch16_224.mae', 'architecture': 'vit_base_patch16_224', 'tag': 'mae', 'custom_load': False, 'input_size': (3, 224, 224), 'fixed_input_size': True, 'interpolation': 'bicubic', 'crop_pct': 0.9, 'crop_mode': 'center', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'num_classes': 0, 'pool_size': None, 'first_conv': 'patch_embed.proj', 'classifier': 'head', 'license': 'cc-by-nc-4.0'}
    timm_data_cfg = timm.data.resolve_data_config(pretrained_cfg)
    no_aug_vision_transform = timm.data.create_transform(**timm_data_cfg)
    if args.dataset_cfg.vision_aug:
        timm_data_cfg["is_training"] = True
        timm_data_cfg["hflip"] = 0.0
        timm_data_cfg["scale"] = (0.65, 1.0)
        timm_data_cfg["ratio"] = (1.0, 1.0)
    vision_transform = timm.data.create_transform(**timm_data_cfg)
    return vision_transform, no_aug_vision_transform

def get_data_from_h5(data, episode_name, resolution=(224, 224, 3), return_PIL_images=False):
    # img, action and proprio keys can be found in config/dataset_config_template.yaml
    # side_images = data[f"{episode_name}/observation/rgb_static"][:]
    # wrist_images = data[f"{episode_name}/observation/rgb_gripper"][:]
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
    if initial_state is not None:
        obs = env.reset(**initial_state)
    else:
        obs = env.get_obs()
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
        obs = preprocess_obs(obs)
        side_image = obs["rgb_obs"]["rgb_static"].astype(np.uint8)
        wrist_image = obs["rgb_obs"]["rgb_gripper"].astype(np.uint8)
        if not isinstance(side_image, Image.Image):
            if side_image.shape[0] == 3:
                side_image = side_image.transpose(1, 2, 0)
                wrist_image = wrist_image.transpose(1, 2, 0)
            side_image = Image.fromarray(side_image)
            wrist_image = Image.fromarray(wrist_image)
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
        '''

        if use_gt:
            action = gt_actions[step]
        else:
            action = pred_action

        # this loss will be much higher if pred_action is not using ground truth observations as the trajectories are a bit different
        loss_val = 0.0
        if gt_actions is not None:
            if (step < len(gt_actions)):
                loss_val = loss(torch.tensor(pred_action), torch.tensor(gt_actions[step])).item()
            else:
                loss_val = loss(torch.tensor(pred_action), torch.tensor(gt_actions[-1])).item()
        avg_loss += loss_val
        # print(loss_val)

        print(f"Step: {step}, Loss: {loss_val:.3f}, action: {action}")
        # import ipdb; ipdb.set_trace()
        action[-1] = 1 if action[-1] > 0 else -1
        obs, _, _, current_info = env.step(action)
        # import ipdb; ipdb.set_trace()
        if args.debug:
            img = env.render()
            time.sleep(0.05)

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

def evaluate_policy_singlestep(env, icrt, prompt_dict, args):
    # conf_dir = Path(__file__).absolute().parents[2] / "conf"
    dataroot = args.dataroot
    conf_dir = get_conf_path()
    task_cfg = OmegaConf.load(os.path.join(conf_dir, "callbacks/rollout/tasks/new_playtable_tasks.yaml"))
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(os.path.join(conf_dir, "annotations/new_playtable_validation.yaml"))
    lang_annotations = np.load(os.path.join(dataroot, "lang_annotations", "auto_lang_ann.npy"), allow_pickle=True).item()
    task_name = args.task_name

    results = Counter()

    total_trajs = 0
    # task_to_id_dict = task_oracle.task_to_id
    global_avg_loss = 0.0

    print(f"Task: {task_name}")
    # episodes_names = [f"task_{task_name}_episode_{i}" for i in range(min(len(data.keys()), 10))]
    # traj_list = [get_data_from_h5(data, episode_name, return_PIL_images=True) for episode_name in episodes_names]
    traj_list = get_task_to_id_dict(dataroot, task_name, lang_annotations)
    for traj in traj_list:
        print(f"Task: {task_name}")
        total_trajs += 1
        obs, proprio, actions, init_state = traj
        actions = np.concatenate([actions[0], actions[1]], axis=1)
        icrt.reset()
        icrt.prompt(**prompt_dict)
        success, avg_loss = rollout(
            icrt=icrt,
            env=env,
            initial_state=init_state,
            task_oracle=task_oracle,
            task=task_name,
            args=args,
            gt_actions=actions,
            gt_obs=obs,
            gt_proprio=proprio,
            use_gt=args.use_gt
        )
        global_avg_loss += avg_loss
        print(colored(f"Success: {success}, Avg Loss: {avg_loss:.3f}", "yellow"))
        if success:
            results[task_name] += 1
        # visualize_trajectory(obs)

    print(f"Total eval trajs: {total_trajs}")
    print(f"Global Avg Loss: {global_avg_loss / total_trajs:.3f}")
    print(f"SR: {sum(results.values()) / total_trajs * 100:.1f}%")
    return

def evaluate_sequence(env, icrt, task_oracle, initial_state, eval_sequence, val_annotations, plans, args):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if args.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        # find the correct prompt for the subtask
        print(subtask, end=" ")
        task_dataset_path = [path for path in args.dataset_paths if subtask in path]
        print(task_dataset_path)
        assert len(task_dataset_path) == 1, f"Task dataset path not found for {subtask}. Task dataset path: {task_dataset_path}"
        data = load_hdf5(task_dataset_path[0])
        prompt_side_images, prompt_wrist_images, prompt_proprios, prompt_actions = get_prompt(data, subtask, args.n_prompt)
        # visualize_trajectory(prompt_side_images)
        prompt_dict = {
            "side_image": prompt_side_images,
            "wrist_image": prompt_wrist_images,
            "proprio": prompt_proprios,
            "action": prompt_actions
        }
        icrt.reset()
        icrt.prompt(**prompt_dict)
        success = rollout(
            icrt=icrt,
            env=env,
            initial_state=None, # we pass it as None so that we do not reset the environment for each subtask
            task_oracle=task_oracle,
            task=subtask,
            args=args,
        )
            #val_annotations, plans, debug)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def evaluate_policy(icrt, env, args):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = get_conf_path()
    task_cfg = OmegaConf.load(os.path.join(conf_dir, "callbacks/rollout/tasks/new_playtable_tasks.yaml"))
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(os.path.join(conf_dir,"annotations/new_playtable_validation.yaml"))

    NUM_SEQUENCES = 5
    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    plans = defaultdict(list)

    if not args.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env=env,
            icrt=icrt,
            task_oracle=task_oracle,
            initial_state=initial_state,
            eval_sequence=eval_sequence,
            val_annotations=val_annotations,
            plans=plans, args=args,
        )
        results.append(result)
        if not args.debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    return results


def main(args):
    # Load the config file
    with open(args.cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    dataset_paths = cfg['dataset_path']
    args.dataset_paths = dataset_paths
    print(colored(f"Dataset paths: {dataset_paths}", "yellow"))
    task_name = args.task_name
    dataset_paths = []
    if task_name is not None:
        dataset_paths = [path for path in dataset_paths if task_name in path]

    checkpoint_path = args.ckpt_path
    train_yaml_path = args.train_yaml_path
    vision_encoder_path = args.vision_encoder_path # if args.vision_encoder_path is not None else '/home/rutavms/research/gaze/icrt/checkpoints/crossmae_rtx/cross-mae-rtx-vitb.pth'
    icrt = ICRTWrapper(train_yaml_path, checkpoint_path, vision_encoder_path)
    # icrt.model = torch.compile(icrt.model)
    resolution = (224, 224, 3)

    if False:
        val_only = False
        train_only = True

        # load the run.yaml file
        # run_cfg = EasyDict(yaml.safe_load(open(train_yaml_path, 'r')))
        run_cfg = yaml.load(Path(train_yaml_path).read_text(), Loader=yaml.Loader)
        vision_transform, no_aug_vision_transform = load_vision_transforms(run_cfg)
        dataset_train, dataset_val = load_datasets(run_cfg, vision_transform, no_aug_vision_transform, val_only=val_only, train_only=train_only)
        avg_loss = 0.0
        index = 0
        avg_time = 0.0
        compile_index = 0
        for dataset_item in dataset_train:
            # add a batch dimension
            for k, v in dataset_item.items():
                dataset_item[k] = v.unsqueeze(0).to('cuda', non_blocking=True).contiguous()
                # make the memory contiguous

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                start_time = time.time()
                loss, loss_dict = icrt.model(dataset_item)
                end_time = time.time()
                avg_loss += loss.item()
                if index > 5: # skip the first 5 iterations
                    avg_time += end_time - start_time
                    compile_index += 1
            print(f"Running average loss: {avg_loss / (index + 1)}")
            # if compile_index > 0:
            #     print(f"Avg time per iteration: {avg_time / compile_index}")
            index += 1
            if index == 50:
                break
        # exit()


    # Load the dataset
    if False:
        assert args.task_name is not None, "Task name must be provided for evaluation of single step task"
        assert len(dataset_paths) > 0, f"No dataset paths found for task {task_name} in {args.mode} mode. Exiting..."
        raise NotImplementedError("Single step evaluation check the dataset paths argument here")
        data = load_hdf5(dataset_paths[0]) # only 0th dataset for now
        prompt_side_images, prompt_wrist_images, prompt_proprios, prompt_actions = get_prompt(data, task_name, args.n_prompt)

        total_loss = 0
        loss_calc_index = 0
        for index in range(0, len(data.keys())):
            print(f"Episode: {index}")
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
            index = 0
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
                loss_avg += loss.item()
                index += 1

            loss_avg /= index
            total_loss += loss_avg
            loss_calc_index += 1
            print(f"Average L1 loss: {loss_avg:.4f}")
            print(f"Total Average L1 loss: {total_loss / loss_calc_index:.4f}")
        print(f"Total Average L1 loss: {total_loss / loss_calc_index:.4f}")
        exit()

    datamodule = None
    data_name = "task_D_D"
    dataroot = os.path.join(os.environ["CALVIN_DATAROOT"], data_name, 'training' if args.mode=='train' else 'validation')
    args.dataroot = dataroot
    # env, datamodule, dataroot = get_env_datamodule(args)
    obs_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["actions"],
        "language": ["language"]
    }
    if False: # this is the MTLC baseline. There could be bias in initial conditions and the task
        env, datamodule, dataroot = get_env_datamodule(args)
        env.reset()
        # env.reset()

        # Evaluate the model
        prompt_dict = {
            "side_image": prompt_side_images,
            "wrist_image": prompt_wrist_images,
            "proprio": prompt_proprios,
            "action": prompt_actions
        }
        evaluate_policy_singlestep(env, icrt, prompt_dict, args)

    if True:
        # env = make_env(dataroot, obs_space=obs_space, show_gui=False)
        env, datamodule, dataroot = get_env_datamodule(args)
        results = evaluate_policy(icrt, env, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--train_yaml_path", type=str, default=None)
    parser.add_argument("--cfg_path", type=str, required=True, help="path to the config file with the dataset paths")
    parser.add_argument("--vision_encoder_path", type=str, default=None)
    parser.add_argument("--task_name", type=str, required=False)
    parser.add_argument("--n_prompt", type=int, default=1)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_gt", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.train_yaml_path is None:
        args.train_yaml_path = '/'.join(args.ckpt_path.split('/')[:-1] + ['run.yaml'])
    seed_everything(args.seed, workers=True)
    args.ep_len = 150

    main(args)
