import h5py
from tqdm import tqdm
from pathlib import Path
import datetime
import json
import numpy as np
import os
import time
import tyro
import wandb
import yaml

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.data.loader import MultiEpochsDataLoader

from icrt.data import load_datasets
from icrt.data.dataset import SequenceDataset, PlayDataset

import icrt.util.misc as misc
from icrt.util.misc import NativeScalerWithGradNormCount as NativeScaler
from icrt.util.args import ExperimentConfig
from icrt.util.engine import train_one_epoch
from icrt.util.model_constructor import model_constructor

def main(args : ExperimentConfig):

    # Loading data config

    # hdf5_file = h5py.File(data_cfg['dataset_path'][0], 'r')
    # print("Keys: %s" % hdf5_file.keys())
    # hdf5_file.keys(): episode names
    # hdf5_file['episode_0'].keys():
    # ['action', 'discount', 'is_first', 'is_last', 'is_terminal', 'language_embedding', 'language_embedding_2', 'language_embedding_3', 'language_instruction', 'language_instruction_2', 'language_instruction_3', 'observation', 'reward']>
    # hdf5_file['episode_0']['observation'].keys(): ['cartesian_position', 'exterior_image_1_left', 'exterior_image_2_left', 'gripper_position', 'joint_position', 'wrist_image_left']>
    # hdf5_file['episode_0']['action'].keys(): ['cartesian_position', 'cartesian_velocity', 'gripper_position', 'gripper_velocity', 'joint_position', 'joint_velocity']>

    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.shared_cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    for dataset_json in args.dataset_cfg.dataset_json:
        data_cfg = json.load(open(dataset_json, 'r'))
        # make sure the number of cameras is correct
        rgb_observations = data_cfg["image_keys"]
        assert len(rgb_observations) == args.shared_cfg.num_cameras, "Number of cameras must match the number of rgb observations"

    eff_batch_size = args.shared_cfg.batch_size * args.trainer_cfg.accum_iter * misc.get_world_size()

    if args.optimizer_cfg.lr is None:  # only base_lr is specified
        args.optimizer_cfg.lr = args.optimizer_cfg.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.optimizer_cfg.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.optimizer_cfg.lr)

    print("accumulate grad iterations: %d" % args.trainer_cfg.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # model = model_constructor(
    #     model_config=args.model_cfg,
    #     shared_config=args.shared_cfg,
    #     train=args.train,
    # )
    pretrained_cfg = {'url': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth', 'hf_hub_id': 'timm/vit_base_patch16_224.mae', 'architecture': 'vit_base_patch16_224', 'tag': 'mae', 'custom_load': False, 'input_size': (3, 224, 224), 'fixed_input_size': True, 'interpolation': 'bicubic', 'crop_pct': 0.9, 'crop_mode': 'center', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'num_classes': 0, 'pool_size': None, 'first_conv': 'patch_embed.proj', 'classifier': 'head', 'license': 'cc-by-nc-4.0'}
    timm_data_cfg = timm.data.resolve_data_config(pretrained_cfg)
    no_aug_vision_transform = timm.data.create_transform(**timm_data_cfg)
    if args.dataset_cfg.vision_aug:
        timm_data_cfg["is_training"] = True
        timm_data_cfg["hflip"] = 0.0
        timm_data_cfg["scale"] = (0.65, 1.0)
        timm_data_cfg["ratio"] = (1.0, 1.0)
    vision_transform = timm.data.create_transform(**timm_data_cfg)
    # following timm: set wd as 0 for bias and norm layers
    # param_groups = misc.add_weight_decay(model_without_ddp, args.optimizer_cfg.weight_decay)
    # optimizer = torch.optim.AdamW(param_groups, lr=args.optimizer_cfg.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler()

    dataset_train, dataset_val = load_datasets(args, vision_transform, no_aug_vision_transform)

    # reverse the order of the dataset
    # for data in reversed(dataset_train):
        # a = 2
        # data: observation, proprio, action, prompt_mask, weight_mask
        # data['observation']: (512, 2, 3, 224, 224): (seq_length, num_cameras, C, H, W)
        # data['proprio']: (512, 16, 10): (seq_length, timesteps, proprio_dim)
        # data['action']: (512, 16, 11): (seq_length, timesteps, action_dim)
        # data['prompt_mask']: (512) # this tells us what is part of the prompt and what is not; 1 if not part of prompt, 0 if part of prompt
        # data['weight_mask']: (512)

    for data in tqdm(dataset_train):
        # import ipdb; ipdb.set_trace()
        a = 2


if __name__ == '__main__':
    # parsing args
    args = tyro.cli(ExperimentConfig)

    if args.load_config is not None:
        print("loading configs from file: ", args.load_config)
        assert os.path.exists(args.load_config), f"Config file does not exist: {args.load_config}"
        args : ExperimentConfig = yaml.load(Path(args.load_config).read_text(), Loader=yaml.Loader)

    # creating the output directory and logging directory
    if args.logging_cfg.log_name is not None:
        args.logging_cfg.output_dir = os.path.join(args.logging_cfg.output_dir, args.logging_cfg.log_name)
    if args.logging_cfg.log_dir is None:
        args.logging_cfg.log_dir = args.logging_cfg.output_dir
    if args.logging_cfg.output_dir:
        Path(args.logging_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # dump the args into a yaml file
    with open(os.path.join(args.logging_cfg.output_dir, "run.yaml"), 'w') as f:
        yaml.dump(args, f)

    main(args)
