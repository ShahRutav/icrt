import os
import h5py
import json
import argparse
import numpy as np

from icrt.util.calvin_utils import generate_dataset_paths, get_subsequence, break_subsequence, taskname2taskname

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def main(args):
    dataset_config = {
        "dataset_path": [],
        "hdf5_keys": [],
        "epi_len_mapping_json": "",
        "verb_to_episode": "",
        "action_keys": ["action/cartesian_pose", "action/gripper_position"],
        "image_keys": ["observation/rgb_static", "observation/rgb_gripper"],
        "proprio_keys": ["observation/cartesian_pose", "observation/gripper_position"],
    }
    dataset_config["train_split"] = 1.0 if args.split == "train" else 0.0

    dataroot = os.environ["CALVIN_DATAROOT"]
    dataroot = os.path.join(dataroot, 'calvin_' + args.benchmark_name + '_dataset', "training" if args.split == "train" else "validation")

    annotations = np.load(os.path.join(dataroot, "lang_annotations", "auto_lang_ann.npy"), allow_pickle=True).item()
    # ['language']['ann']: list of raw language
    # ['language']['task']: list of task_id
    # ['language']['emb']: precomputed miniLM language embedding
    # ['info']['indx']: list of start and end indices corresponding to the precomputed language embeddings
    task_ids = []
    for i, ((start, end), ann) in enumerate(zip(annotations['info']['indx'], annotations['language']['task'])):
        ann = taskname2taskname(ann)
        print(f"Episode {i}: {ann}")
        print(f"Start: {start}, End: {end}")
        task_ids.append(ann)

    # make task_ids unique
    task_ids = list(set(task_ids))
    # each task_id will be a unique hdf5 dataset
    print(f"Unique task_ids: {task_ids}")
    combined_episode_length_map = {}
    comb_ep_len_json_path = os.path.join(dataroot, args.output_name, 'combined_epi_len_mapping.json')
    comb_verb_to_ep_map = {}
    comb_verb_to_ep_map_path = os.path.join(dataroot, args.output_name, 'verb_to_episode.json')
    for task_id in task_ids:
        demo_path = os.path.join(dataroot, args.output_name, task_id + ".hdf5")
        os.makedirs(os.path.dirname(demo_path), exist_ok=True)

        f_out = h5py.File(demo_path, 'w')
        data_grp = f_out
        episode_length_json = {}
        i = 0

        for _i, ((start, end), ann) in enumerate(zip(annotations['info']['indx'], annotations['language']['task'])):
            ann = taskname2taskname(ann)
            if ann != task_id:
                continue
            # generate dataset paths
            dataset_paths = generate_dataset_paths(dataroot, start, end)
            # get subsequence
            trajectory = get_subsequence(dataset_paths)
            # break subsequence
            actions, proprio, obs = break_subsequence(trajectory)

            grp_name = f"task_{task_id}_episode_{i}"
            ep_data_grp = data_grp.create_group(grp_name)
            ep_data_grp.create_dataset("action/cartesian_pose", data=actions[0])
            ep_data_grp.create_dataset("action/gripper_position", data=actions[1])
            ep_data_grp.create_dataset("observation/cartesian_pose", data=proprio[0])
            ep_data_grp.create_dataset("observation/gripper_position", data=proprio[1])
            ep_data_grp.create_dataset("observation/rgb_gripper", data=obs["rgb_gripper"])
            ep_data_grp.create_dataset("observation/rgb_static", data=obs["rgb_static"])

            ep_data_grp.attrs["num_samples"] = actions[0].shape[0] # number of transitions in this episode
            episode_length_json[grp_name] = actions[0].shape[0] # number of transitions in this episode
            i += 1


        # # save the json file by the same name as the output file but with .json extension
        # json_output_path = demo_path[:-5] + '.json'
        # with open(json_output_path, 'w') as f:
        #     json.dump(episode_length_json, f, cls=NpEncoder)

        json_output_path = demo_path[:-5] + '_keys.json'
        keys = list(episode_length_json.keys())
        with open(json_output_path, 'w') as f:
            json.dump(keys, f, cls=NpEncoder, indent=4)

        dataset_config["dataset_path"].append(demo_path)
        dataset_config["hdf5_keys"].append(json_output_path)

        # extend the combined_episode_length_map with the episode_length_json
        combined_episode_length_map.update(episode_length_json)
        comb_verb_to_ep_map[task_id] = keys

    dataset_config["epi_len_mapping_json"] = comb_ep_len_json_path
    dataset_config["verb_to_episode"] = comb_verb_to_ep_map_path

    # save the combined_episode_length_map
    with open(comb_ep_len_json_path, 'w') as f:
        json.dump(combined_episode_length_map, f, cls=NpEncoder, indent=4)

    # save the combined_verb_to_ep_map
    with open(comb_verb_to_ep_map_path, 'w') as f:
        json.dump(comb_verb_to_ep_map, f, cls=NpEncoder, indent=4)

    # save the dataset_config
    dataset_config_path = os.path.join(dataroot, args.output_name, 'dataset_config.json')
    with open(dataset_config_path, 'w') as f:
        json.dump(dataset_config, f, indent=4, cls=NpEncoder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark_name", "-b",
        type=str,
        required=True,
        help="name of benchmark to use",
        # generate a list of possible benchmarks
        choices=["debug", "D", "ABC", "ABCD"],

    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="train or val",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=200,
        help="(optional) height of image observations",
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=200,
        help="(optional) width of image observations",
    )
    # strategy: default is task_id
    parser.add_argument(
        "--strategy",
        type=str,
        default="task_id",
        help="strategy for splitting data",
    )
    args = parser.parse_args()
    main(args)
