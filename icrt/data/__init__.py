import torch
from .dataset import PlayDataset, SequenceDataset, ConcatDatasetO

def get_dataset(args, dataset_kwargs):
    dataset_train, dataset_val = None, None
    if ('calvin' in args.dataset_cfg.dataset_json) and ('group' not in args.dataset_cfg.dataset_json):
        dataset_train = PlayDataset(
            split="train",
            **dataset_kwargs
        )
        dataset_val = PlayDataset(
            split="val",
            **dataset_kwargs
        )
    else:
        dataset_train = SequenceDataset(
            split="train",
            **dataset_kwargs
        )
        dataset_val = SequenceDataset(
            split="val",
            **dataset_kwargs
        )
    return dataset_train, dataset_val

def load_datasets(args, vision_transform, no_aug_vision_transform):
    assert len(args.dataset_cfg.dataset_json) == len(args.dataset_cfg.dataset_val_json), "Number of train and val datasets should be the same: {} != {}".format(len(args.dataset_cfg.dataset_json), len(args.dataset_cfg.dataset_val_json))
    assert len(args.dataset_cfg.dataset_json) == len(args.dataset_cfg.num_repeat_traj), "Number of train datasets and num_repeat_traj should be the same: {} != {}".format(len(args.dataset_cfg.dataset_json), len(args.dataset_cfg.num_repeat_traj))
    if len(args.dataset_cfg.dataset_json) == 1:
        args.dataset_cfg.dataset_json = args.dataset_cfg.dataset_json[0]
        args.dataset_cfg.dataset_val_json = args.dataset_cfg.dataset_val_json[0]
        args.dataset_cfg.num_repeat_traj = args.dataset_cfg.num_repeat_traj[0]
        print("*"*20)
        print(args.dataset_cfg.dataset_json)
        dataset_kwargs = {
            "shared_config": args.shared_cfg,
            "dataset_config": args.dataset_cfg,
            "vision_transform": vision_transform,
            "no_aug_vision_transform": no_aug_vision_transform,
        }
        dataset_train, dataset_val = get_dataset(args, dataset_kwargs)
        print("Length of dataset_train: ", len(dataset_train))
        print("Length of dataset_val: ", len(dataset_val))
        print("*"*20)
    else:
        dataset_train_list, dataset_val_list = [], []
        dataset_jsons = args.dataset_cfg.dataset_json
        dataset_val_json = args.dataset_cfg.dataset_val_json
        num_repeat_trajs = args.dataset_cfg.num_repeat_traj
        for dataset_json, dataset_val_json, num_repeat_traj in zip(dataset_jsons, dataset_val_json, num_repeat_trajs):
            args.dataset_cfg.dataset_json = dataset_json
            args.dataset_cfg.dataset_val_json = dataset_val_json
            args.dataset_cfg.num_repeat_traj = num_repeat_traj
            print("*"*20)
            print(dataset_json)
            dataset_kwargs = {
                "shared_config": args.shared_cfg,
                "dataset_config": args.dataset_cfg,
                "vision_transform": vision_transform,
                "no_aug_vision_transform": no_aug_vision_transform,
            }
            _dataset_train, _dataset_val = get_dataset(args, dataset_kwargs)
            print("Length of dataset_train: ", len(_dataset_train))
            print("Length of dataset_val: ", len(_dataset_val))
            print("*"*20)
            dataset_train_list.append(_dataset_train)
            dataset_val_list.append(_dataset_val)
        dataset_train = ConcatDatasetO(dataset_train_list)
        dataset_val = ConcatDatasetO(dataset_val_list)

        print("*"*20)
        print("Length of dataset_train: ", len(dataset_train))
        print("Length of dataset_val: ", len(dataset_val))
    return dataset_train, dataset_val

