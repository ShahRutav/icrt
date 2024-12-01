import torch
from termcolor import colored
from .dataset import PlayDataset, SequenceDataset, CustomConcatDataset

def get_dataset(args, dataset_kwargs, val_only=False, train_only=False):
    dataset_train, dataset_val = None, None
    # if ('calvin' in args.dataset_cfg.dataset_json) and ('group' not in args.dataset_cfg.dataset_json):
    if 'play' in args.dataset_cfg.dataset_json:
        # print in red that we are using PlayDataset
        print(colored("Using PlayDataset", "red"))
        dataset_train = []
        if not val_only:
            dataset_train = PlayDataset(
                split="train",
                **dataset_kwargs
            )
        dataset_val = []
        if not train_only:
            dataset_val = PlayDataset(
                split="val",
                **dataset_kwargs
            )
    else:
        print(colored("Using SequenceDataset", "red"))
        dataset_train = []
        if not val_only:
            dataset_train = SequenceDataset(
                split="train",
                **dataset_kwargs
            )
        dataset_val = []
        if not train_only:
            dataset_val = SequenceDataset(
                split="val",
                **dataset_kwargs
            )
    return dataset_train, dataset_val

def load_datasets(args, vision_transform, no_aug_vision_transform, val_only=False, train_only=False):
    assert len(args.dataset_cfg.dataset_json) == len(args.dataset_cfg.dataset_val_json), "Number of train and val datasets should be the same: {} != {}".format(len(args.dataset_cfg.dataset_json), len(args.dataset_cfg.dataset_val_json))
    assert len(args.dataset_cfg.dataset_json) == len(args.dataset_cfg.num_repeat_traj), "Number of train datasets and num_repeat_traj should be the same: {} != {}".format(len(args.dataset_cfg.dataset_json), len(args.dataset_cfg.num_repeat_traj))
    assert len(args.dataset_cfg.dataset_json) == len(args.dataset_cfg.non_overlapping), "Number of train datasets and non_overlapping should be the same: {} != {}".format(len(args.dataset_cfg.dataset_json), len(args.dataset_cfg.non_overlapping))
    if len(args.dataset_cfg.dataset_json) == 1:
        args.dataset_cfg.dataset_json = args.dataset_cfg.dataset_json[0]
        args.dataset_cfg.dataset_val_json = args.dataset_cfg.dataset_val_json[0]
        args.dataset_cfg.num_repeat_traj = args.dataset_cfg.num_repeat_traj[0]
        args.dataset_cfg.non_overlapping = args.dataset_cfg.non_overlapping[0]
        print("*"*20)
        print(args.dataset_cfg.dataset_json)
        dataset_kwargs = {
            "shared_config": args.shared_cfg,
            "dataset_config": args.dataset_cfg,
            "vision_transform": vision_transform,
            "no_aug_vision_transform": no_aug_vision_transform,
        }
        dataset_train, dataset_val = get_dataset(args, dataset_kwargs, val_only, train_only)
        print("Length of dataset_train: ", len(dataset_train))
        print("Length of dataset_val: ", len(dataset_val))
        print("*"*20)
    else:
        dataset_train_list, dataset_val_list = [], []
        dataset_jsons = args.dataset_cfg.dataset_json
        dataset_val_json = args.dataset_cfg.dataset_val_json
        num_repeat_trajs = args.dataset_cfg.num_repeat_traj
        non_overlapping = args.dataset_cfg.non_overlapping
        for dataset_index, (dataset_json, dataset_val_json) in enumerate(zip(dataset_jsons, dataset_val_json)):
            args.dataset_cfg.dataset_json = dataset_json
            args.dataset_cfg.dataset_val_json = dataset_val_json
            args.dataset_cfg.num_repeat_traj = num_repeat_trajs[dataset_index]
            args.dataset_cfg.non_overlapping = non_overlapping[dataset_index]
            print("*"*20)
            print(dataset_json)
            dataset_kwargs = {
                "shared_config": args.shared_cfg,
                "dataset_config": args.dataset_cfg,
                "vision_transform": vision_transform,
                "no_aug_vision_transform": no_aug_vision_transform,
            }
            _dataset_train, _dataset_val = get_dataset(args, dataset_kwargs, val_only, train_only)
            print("Length of dataset_train: ", len(_dataset_train))
            print("Length of dataset_val: ", len(_dataset_val))
            print("*"*20)
            dataset_train_list.append(_dataset_train)
            dataset_val_list.append(_dataset_val)
        if not val_only:
            dataset_train = CustomConcatDataset(dataset_train_list)
        if not train_only:
            dataset_val = CustomConcatDataset(dataset_val_list)

        print("*"*20)
        print("Length of dataset_train: ", len(dataset_train))
        print("Length of dataset_val: ", len(dataset_val))
    return dataset_train, dataset_val

