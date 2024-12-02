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
# dinov2 path: /home/rutavms/research/gaze/Lotus/lotus/skill_learning/dinov2
sys.path.append("/home/rutavms/research/gaze/Lotus/lotus/skill_learning/dinov2")


from einops import rearrange
from easydict import EasyDict
import torch.backends.cudnn as cudnn

from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup

from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import get_autocast_dtype, build_model_for_eval
from dinov2.eval.utils import ModelWithIntermediateLayers

from sklearn.decomposition import PCA

from icrt.util.calvin_utils import generate_dataset_paths, get_subsequence, break_subsequence, load_obs, generate_dataset_paths_from_indices, load_action

USE_GPU = torch.cuda.is_available()
def safe_cuda(x):
    if USE_GPU:
        return x.cuda()
    return x

def process_batch_images(images, dinov2):
    resized_images = [cv2.resize(img, (448, 448), interpolation=cv2.INTER_NEAREST) for img in images]
    features = dinov2.process_images(resized_images)
    features_batch = rescale_feature_map(torch.as_tensor(features).permute(0, 3, 1, 2), 1, 1, convert_to_numpy=False).squeeze()  # (B, 768)
    if features_batch.shape[0] == 1:
        features_batch = features_batch[None]
    return features_batch

class DinoV2ImageProcessor(object):
    def __init__(self, args=None):
        if args is None:
            self.args = EasyDict()
            self.args.output_dir = ''
            self.args.opts = []
            self.args.pretrained_weights = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
            self.args.config_file = "/home/rutavms/research/gaze/Lotus/lotus/skill_learning/dinov2/dinov2/configs/eval/vitb14_pretrain.yaml"
        else:
            self.args = args
        # print("*****")
        print(self.args)
        self.model, self.autocast_dtype = self.setup_and_build_model()
        self.n_last_blocks_list = [1, 4]
        self.n_last_blocks = max(self.n_last_blocks_list)
        self.autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=self.autocast_dtype)
        self.feature_model = ModelWithIntermediateLayers(self.model, self.n_last_blocks, self.autocast_ctx)

    @staticmethod
    def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
        return x

    def setup_and_build_model(self):
        cudnn.benchmark = True
        config = setup(self.args)
        model = build_model_for_eval(config, self.args.pretrained_weights)
        model.eval()
        autocast_dtype = get_autocast_dtype(config)
        return model, autocast_dtype

    def process_image(self, img):
        # img = cv2.imread(image_path)
        sizes = [448, 224]
        features = []
        max_size = max(sizes) // 14

        for size in sizes:
            img = cv2.resize(img, (size, size))
            img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.
            img_tensor = self.color_normalize(img_tensor)
            feature = self.feature_model(img_tensor)[-1][0]
            new_feat = torch.nn.functional.interpolate(rearrange(feature, 'b (h w) c -> b c h w', h=int(np.sqrt(feature.shape[1]))), (max_size, max_size), mode="bilinear", align_corners=True, antialias=True)
            new_feat = rearrange(new_feat, 'b c h w -> b h w c')
            features.append(new_feat.squeeze(0))

        features = torch.mean(torch.stack(features), dim=0)
        return features
        # return self.pca_transform(features, max_size)

    def process_images(self, imgs):
        # imgs should be a batch of images, shape (batch_size, height, width, channels)
        sizes = [448, 224]
        max_size = max(sizes) // 14
        batch_size = len(imgs)

        all_features = []
        for size in sizes:
            imgs_resized = [cv2.resize(img, (size, size)) for img in imgs]
            img_tensors = torch.stack([torch.tensor(img).permute(2, 0, 1).float() for img in imgs_resized]).cuda() / 255.
            img_tensors = torch.cat([self.color_normalize(img_tensor.unsqueeze(0)) for img_tensor in img_tensors])
            features = self.feature_model(img_tensors)[-1][0]
            new_feats = torch.nn.functional.interpolate(rearrange(features, 'b (h w) c -> b c h w', h=int(np.sqrt(features.shape[1]))), (max_size, max_size), mode="bilinear", align_corners=True, antialias=True)
            new_feats = rearrange(new_feats, 'b c h w -> b h w c')
            all_features.append(new_feats)

        all_features = torch.mean(torch.stack(all_features), dim=0)
        return all_features


    @staticmethod
    def pca_transform(features, max_size):
        pca = PCA(n_components=3)
        pca_tensor = pca.fit_transform(features.detach().cpu().numpy().reshape(-1, 768))
        pca_tensor = (pca_tensor - pca_tensor.min()) / (pca_tensor.max() - pca_tensor.min())
        pca_tensor = (pca_tensor * 255).astype(np.uint8).reshape(max_size, max_size, 3)
        # pca_tensor = 2 * pca_tensor - 1
        pca_tensor = pca_tensor.reshape(max_size, max_size, 32)
        return pca_tensor

    def save_image(self, pca_tensor, out_path="dinov2_pca.png"):
        cv2.imwrite(out_path, pca_tensor)

def compute_affinity(feat_1_tuple, feat_2_tuple, temperature=1):
    feat_1, h, w = feat_1_tuple
    feat_2, h2, w2 = feat_2_tuple
    feat_1 = rearrange(feat_1, 'h w c -> (h w) c')
    feat_2 = rearrange(feat_2, 'h w c -> (h w) c')
    sim_matrix = torch.einsum("lc,sc->ls", feat_1, feat_2) / temperature
    aff = sim_matrix
    # aff = F.softmax(aff, dim=0)
    aff = aff.cpu().view(h, w, h2, w2)
    # compute softmax over the first two axes
    return aff

def rescale_feature_map(img_tensor, target_h, target_w, convert_to_numpy=True):
    img_tensor = torch.nn.functional.interpolate(img_tensor, (target_h, target_w))
    if convert_to_numpy:
        return img_tensor.cpu().numpy()
    else:
        return img_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-name',
        type=str,
        default="debug",
    )
    parser.add_argument(
        '--benchmark-name', '-b',
        type=str,
        default="debug",
    )
    parser.add_argument(
        '--feature-dim',
        type=int,
        default=768*2,
    )
    parser.add_argument(
        '--cfg',
        type=str,
        default="config/dataset_config_calvin.json",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100
    )
    parser.add_argument(
        '--split',
        type=str,
        default="train"
    )
    args = parser.parse_args()

    dataset_json = json.load(open(args.cfg, "r"))
    dataset_json = EasyDict(dataset_json)

    feature_dim = args.feature_dim
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

    index2feature = {'action': {}}

    dinov2 = DinoV2ImageProcessor()

    for batch_idx in trange(0, len(indices), args.batch_size):
        dataset_paths = generate_dataset_paths_from_indices(dataroot, indices[batch_idx:batch_idx+args.batch_size])
        subsequence = get_subsequence(dataset_paths)
        obs = load_obs(subsequence, keys=image_keys)
        actions = load_action(subsequence)
        for img_key in image_keys:
            if img_key not in index2feature:
                index2feature[img_key] = {}
            images = obs[img_key]
            features = process_batch_images(images, dinov2)
            for i, idx in enumerate(indices[batch_idx:batch_idx+args.batch_size]):
                index2feature[img_key][idx] = features[i].cpu().numpy()
        for i in range(len(indices[batch_idx:batch_idx+args.batch_size])):
            action_vals = np.concatenate([actions[0], actions[1]], axis=-1)
            index2feature["action"][indices[batch_idx + i]] = action_vals[i]

    # save it in the dataroot folder with model name and feature dim
    save_path = os.path.join(dataroot, f"{args.exp_name}_dinov2_{feature_dim}.hdf5")
    h5py_file = h5py.File(save_path, "w")
    for img_key in image_keys:
        f = h5py_file.create_group(img_key)
        for idx in indices:
            f.create_dataset(f"{idx}", data=index2feature[img_key][idx])
    f = h5py_file.create_group("actions")
    for idx in indices:
        f.create_dataset(f"{idx}", data=index2feature["action"][idx])
    h5py_file.close()

