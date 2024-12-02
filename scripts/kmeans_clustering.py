import yaml
import json
import torch
import numpy as np
import cv2
import argparse
import h5py
import os
from functools import partial
import sys
from tqdm import tqdm, trange
from einops import rearrange
from easydict import EasyDict
import torch.backends.cudnn as cudnn


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from icrt.util.calvin_utils import generate_dataset_paths, get_subsequence, break_subsequence, load_obs, generate_dataset_paths_from_indices, load_action
from icrt.util.eval_utils import EvalLogger, visualize_trajectory

def split_indices(indices, actions, method='changepoint'):
    '''
        split the indices into two parts
        changepoint method is used to split the indices using the last action value as the changepoint or end of continuous sequence

        Args:
            indices: list of indices (N, )
            actions: list of actions (N, 7)
            method: method to split the indices
        Returns:
            split_indices: list of split indices
                (start, end) for each split
    '''
    assert method in ['changepoint']
    prev_min_clip = indices[0]
    post_max_clip = indices[-1] # never updated for efficiency
    theshold = 5 # we add randomly x transitions before and after the changepoint
    if method == 'changepoint':
        split_indices = []
        start = indices[0]
        for i, index in enumerate(indices):
            if (i<len(indices)-1) and (indices[i+1] != 1 + indices[i]): # detects based on jumps in indices
                change_point_start = max(prev_min_clip, start - theshold)
                change_point_end = index
                split_indices.append((change_point_start, change_point_end))
                start = indices[i+1]
                prev_min_clip = index
                continue
            if (i < len(indices)-1) and (not np.allclose(actions[str(indices[i+1])][-1], actions[str(index)][-1])): # detects based on gripper value
                change_point_start = max(prev_min_clip, start - theshold)
                change_point_end = min(index+theshold, post_max_clip)
                split_indices.append((change_point_start, change_point_end))
                start = indices[i+1]
                continue
            elif i == len(indices)-1:
                change_point_start = max(prev_min_clip, start - theshold)
                change_point_end = indices[-1]
                split_indices.append((change_point_start, change_point_end))
                break
    else:
        raise NotImplementedError
    return split_indices

def get_kmeans_feats(feats, action_feats, split_indices, method='average'):
    '''
        get the kmeans features for the split indices
        Args:
            feats: features (N, 1536)
            split_indices: list of split indices
            method: method to get the kmeans features
        Returns:
            kmeans_feats: list of kmeans features
    '''
    assert method in ['average']
    kmeans_feats = []
    for start, end in split_indices:
        batch_feats = [feats[str(i)][()][None] for i in range(start, end+1)]
        act_batch_feats = [action_feats[str(i)][()][None] for i in range(start, end+1)]
        batch_feats = np.concatenate(batch_feats, axis=0)
        act_batch_feats = np.concatenate(act_batch_feats, axis=0)
        assert len(batch_feats.shape)== 2
        assert len(act_batch_feats.shape) == 2
        mean_batch_feats = np.mean(batch_feats, axis=0)
        f = mean_batch_feats
        # sample 10 action features evenly spaced
        sample_indices = np.linspace(0, act_batch_feats.shape[0]-1, 10).astype(int)
        act_feats = act_batch_feats[sample_indices]
        assert act_feats.shape[0] == 10, f"Action feats shape: {act_feats.shape}. original shape: {act_batch_feats.shape}"
        # concatenate the action features after flattening them
        f = np.concatenate([f, act_feats[:,:6].reshape(-1)], axis=-1)
        # f = act_feats[:,:6].reshape(-1)
        # import ipdb; ipdb.set_trace()
        kmeans_feats.append(f)
    return kmeans_feats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--feat_file',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--benchmark-name', '-b',
        type=str,
        default="debug",
    )
    parser.add_argument(
        '--keys',
        type=str,
        default=["rgb_static"],
    )
    parser.add_argument(
        '--cfg',
        type=str,
        default="config/dataset_config_calvin.json",
    )
    # number of clusters
    parser.add_argument(
        '--num_clusters',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--split',
        type=str,
        default="val"
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
    )
    parser.add_argument(
        '--save_vid_dir',
        type=str,
        default=None,
    )
    args = parser.parse_args()
    if args.feat_file is None:
        args.feat_file = f"{args.benchmark_name}_dinov2_1536.hdf5"

    dataset_json = json.load(open(args.cfg, "r"))
    dataset_json = EasyDict(dataset_json)
    dataroot = os.environ["CALVIN_DATAROOT"]
    if args.benchmark_name == 'debug':
        dirname =  'calvin_' + args.benchmark_name + '_dataset'
    elif args.benchmark_name == 'D':
        dirname = 'task_D_D'
    else:
        raise NotImplementedError


    dataroot = os.path.join(dataroot, dirname, "training" if args.split == "train" else "validation")
    scene_info_path = os.path.join(dataroot, "scene_info.npy")
    indices = next(iter(np.load(scene_info_path, allow_pickle=True).item().values()))
    indices = list(range(indices[0], indices[1] + 1))
    missing_idx_path = os.path.join(dataroot, "missing_idx.npy")
    missing_idx = next(iter(np.load(missing_idx_path, allow_pickle=True).item().values()))
    indices = set(indices) - set(missing_idx)
    indices = list(sorted(indices))

    image_keys = dataset_json["image_keys"]
    proprio_keys = dataset_json["proprio_keys"]
    action_keys = dataset_json["action_keys"]

    # feature_file is in dataroot
    feature_file = os.path.join(dataroot, args.feat_file)
    f = h5py.File(feature_file, 'r')
    print(f.keys())
    # it consists of f['actions'][index], f['rgb_static'][index]  (features)
    actions = f['actions']
    rgb_static_feats = f['rgb_static']

    split_indices = split_indices(indices, actions)
    # randoomly select some indices and map them to the cluster
    print(split_indices)
    if True:
        # start a tqdm loop and update the bar with the split indices
        # check that each split index intermediate indices are present in indices
        for start, end in tqdm(split_indices):
            # tqdm.write(f"Start: {start}, End: {end}")
            # import ipdb; ipdb.set_trace()
            for i in range(start, end+1):
                assert str(i) in f['actions'].keys(), f"Index {i} not in actions"
                assert str(i) in f['rgb_static'].keys(), f"Index {i} not in rgb_static"
            # print("vibe check passs")
            assert all([i in indices for i in range(start, end+1)]), f"Start: {start}, End: {end} not in indices"


    if args.visualize:
        # select 10 random split indices and visualize the rgb_static images
        random_splits = np.random.choice(len(split_indices), 10)
        for i in random_splits:
            start, end = split_indices[i]
            generate_dataset_paths = generate_dataset_paths_from_indices(dataroot, list(range(start, end+1)))
            subseq = get_subsequence(generate_dataset_paths)
            obs_dict = load_obs(subseq, keys=["rgb_static"])
            visualize_trajectory(obs_dict)


    # print(rgb_static_feats.keys())
    kmeans_feats = get_kmeans_feats(rgb_static_feats, actions, split_indices, method='average')
    # create a dictionary consisting of split index (start, end), kmeans feature, local i
    split_kmeans_dict = {}
    for i, (start, end) in enumerate(split_indices):
        split_kmeans_dict[(start, end)] = kmeans_feats[i]


    # perform kmeans clustering
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(kmeans_feats)
    kmeans_labels = kmeans.labels_
    inertia = kmeans.inertia_ # might be used later for filtering the result

    # find all the split indices that belong to the same cluster
    cluster2split_indices = {}
    for i, label in enumerate(kmeans_labels):
        if label not in cluster2split_indices:
            cluster2split_indices[label] = []
        cluster2split_indices[label].append(split_indices[i])
    # get the subseq, images for each cluster and concatenate them
    for cluster, split_indices in cluster2split_indices.items():
        print(f"Cluster {cluster}")
        full_indices = [list(range(start, end+1)) for start, end in split_indices]
        full_indices = [idx for sublist in full_indices for idx in sublist]
        generate_dataset_paths = generate_dataset_paths_from_indices(dataroot, full_indices)
        subseq = get_subsequence(generate_dataset_paths)
        obs_dict = load_obs(subseq, keys=["rgb_static"])
        rgb_static_images = obs_dict["rgb_static"]
        if args.save_vid_dir is not None:
            os.makedirs(args.save_vid_dir, exist_ok=True)
            save_dir = os.path.join(args.save_vid_dir, f"cluster_{cluster}.mp4")
            # create a video for each cluster
            vid_writer = cv2.VideoWriter(save_dir, cv2.VideoWriter_fourcc(*'mp4v'), 30, (rgb_static_images.shape[2], rgb_static_images.shape[1]))
            for i in range(rgb_static_images.shape[0]):
                vid_writer.write(cv2.cvtColor(rgb_static_images[i], cv2.COLOR_RGB2BGR))
            vid_writer.release()

