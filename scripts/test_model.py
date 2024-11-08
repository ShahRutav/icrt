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
from icrt.data.dataset import SequenceDataset

import icrt.util.misc as misc
from icrt.util.misc import NativeScalerWithGradNormCount as NativeScaler
from icrt.util.args import ExperimentConfig
from icrt.util.engine import train_one_epoch
from icrt.util.model_constructor import model_constructor

def main(args : ExperimentConfig):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.shared_cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Loading data config
    data_cfg = json.load(open(args.dataset_cfg.dataset_json, 'r'))

    # make sure the number of cameras is correct
    rgb_observations = data_cfg["image_keys"]
    assert len(rgb_observations) == args.shared_cfg.num_cameras, "Number of cameras must match the number of rgb observations"

    model = model_constructor(
        model_config=args.model_cfg,
        shared_config=args.shared_cfg,
        train=args.train,
    )


    timm_data_cfg = timm.data.resolve_data_config(model.vision_encoder.model.pretrained_cfg)
    no_aug_vision_transform = timm.data.create_transform(**timm_data_cfg)
    if args.dataset_cfg.vision_aug:
        timm_data_cfg["is_training"] = True
        timm_data_cfg["hflip"] = 0.0
        timm_data_cfg["scale"] = (0.65, 1.0)
        timm_data_cfg["ratio"] = (1.0, 1.0)
    vision_transform = timm.data.create_transform(**timm_data_cfg)

    model.to(device)
    import ipdb; ipdb.set_trace()

    model_without_ddp = model

    # controlled by --model-cfg.policy-cfg.pretrained_path flag
    if args.model_cfg.policy_cfg.pretrained_path is not None:
        print("Finetuning from %s" % args.model_cfg.policy_cfg.pretrained_path)
        misc.load_model(model_without_ddp, args.model_cfg.policy_cfg.pretrained_path)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # training detail
    eff_batch_size = args.shared_cfg.batch_size * args.trainer_cfg.accum_iter * misc.get_world_size()

    if args.optimizer_cfg.lr is None:  # only base_lr is specified
        args.optimizer_cfg.lr = args.optimizer_cfg.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.optimizer_cfg.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.optimizer_cfg.lr)

    print("accumulate grad iterations: %d" % args.trainer_cfg.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.optimizer_cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.optimizer_cfg.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    total, trainable = model_without_ddp.get_total_parameters(), model_without_ddp.get_trainable_parameters()
    print("trainable: ", trainable/1e6, "M")
    print("Total params: ", total/1e6, "M")
    print("percentage trainable: ", trainable / total)
    # print all parameter names that are not trainable
    misc.resume_from_ckpt(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    dataset_train = SequenceDataset(
        dataset_config=args.dataset_cfg,
        shared_config=args.shared_cfg,
        vision_transform=vision_transform,
        no_aug_vision_transform=no_aug_vision_transform,
        split="train",
    )
    dataset_val = SequenceDataset(
        dataset_config=args.dataset_cfg,
        shared_config=args.shared_cfg,
        vision_transform=vision_transform,
        no_aug_vision_transform=no_aug_vision_transform,
        split="val"
    )
    print("Length of dataset_train: ", len(dataset_train))
    print("Length of dataset_val: ", len(dataset_val))

    # save the train the val splits
    dataset_train.save_split(os.path.join(args.logging_cfg.output_dir, "train_split.json"))
    dataset_val.save_split(os.path.join(args.logging_cfg.output_dir, "val_split.json"))

    log_writer = None
    print(f"Start training for {args.trainer_cfg.epochs} epochs")
    start_time = time.time()

    # for resume, we need to instantiate new samplers
    resume_reload = args.shared_cfg.resume is not None

    # for epoch in range(args.shared_cfg.start_epoch, args.trainer_cfg.epochs):


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
