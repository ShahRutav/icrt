import os
import yaml
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
from easydict import EasyDict

import robosuite.utils.transform_utils as T
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
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
    raw_obs_to_tensor_obs,
)
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    NpEncoder,
    compute_flops,
)

from libero.lifelong.main import get_task_embs

def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    change_robot_color=False,
    orig_obs=None,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a
            success state. If 1, done is 1 at the end of each trajectory.
            If 2, do both.
    """
    assert states.shape[0] == actions.shape[0]

    geom_names = deepcopy(env.env.sim.model.geom_names)
    _link_names = []
    for name in geom_names:
        if (name is not None) and (("robot" in name) or ('gripper' in name)):
            _link_names.append(name)

    # load the initial state
    env.reset()
    obs = env.set_init_state(initial_state['states'])
    for _ in range(5):
        env.step(np.zeros((7,)))

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        cartesian_pose=[],
        actions=np.array(actions),
        states=np.array(states),
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):
        # change robot color to create domain mismatch
        if change_robot_color:
            for link_name in _link_names:
                geom_id = env.env.sim.model.geom_name2id(link_name)
                geom_rgba = env.env.sim.model.geom_rgba[geom_id]
                # env.env.sim.model.geom_rgba[geom_id] = [0.68, 0.85, 1.0, 1]  # Red color with full opacity
                env.env.sim.model.geom_rgba[geom_id] = [1.0, 0.0, 0.0, 1]  # Red color with full opacity
                env.env.sim.model.geom_matid[geom_id] = 1


        next_obs, _, _, _ = env.step(actions[t - 1])

        # import matplotlib.pyplot as plt
        # # plt.imshow(orig_obs['agentview_rgb'][0])
        # plt.imshow(next_obs['agentview_image'][::-1])
        # plt.axis('off')
        # plt.savefig('test.png')
        # plt.show()
        # exit()

        cart_pos = obs['robot0_eef_pos']
        cart_quat = obs['robot0_eef_quat']
        cart_rpy = T.mat2euler(T.quat2mat(cart_quat))
        traj["cartesian_pose"].append(np.concatenate([cart_pos, cart_rpy]))
        # infer done signal
        done = False
        # done = 1 at end of trajectory
        done = done or (t == traj_len)
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # check the success of the task
    success = env.check_success()
    print("Success: ", success)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])
    traj["success"] = success

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


def dataset_states_to_obs(args):
    # read the args.cfg path and convert it to easydict
    cfg = EasyDict(yaml.load(open(args.cfg, "r"), Loader=yaml.FullLoader))
    # create environment to use for data processing
    benchmark = get_benchmark(args.benchmark_name)(0)
    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]

    task = benchmark.get_task(args.task_id)
    base_dataset_path = get_libero_path("datasets")
    dataset = h5py.File(os.path.join(base_dataset_path, benchmark.get_task_demonstration(args.task_id)), "r")
    _, shape_meta = get_dataset(
        dataset_path=os.path.join(
            base_dataset_path, benchmark.get_task_demonstration(args.task_id)
        ),
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
    )
    env_args = {
        "bddl_file_name": os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        ),
        "camera_heights": args.camera_height,
        "camera_widths": args.camera_width,
    }
    env = OffScreenRenderEnv(**env_args)
    env.reset()
    env.seed(0)


    # list of all demonstration episodes (sorted in increasing number order)
    demos = list(dataset["data"].keys())

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # output file in same directory as input file
    output_path = os.path.join(base_dataset_path, benchmark.get_task_demonstration(args.task_id)[:-5] + '_' + args.output_name)
    f_out = h5py.File(output_path, "w")
    # data_grp = f_out.create_group("data")
    data_grp = f_out
    print("output file: {}".format(output_path))

    total_samples = 0
    num_success = 0
    episode_length_json = {}
    for ind in range(len(demos)):
        ep = demos[ind]

        # prepare initial state to reload from
        states = dataset["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        # extract obs, rewards, dones
        actions = dataset["data/{}/actions".format(ep)][()]
        traj = extract_trajectory(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            change_robot_color=False,
            orig_obs=dataset["data/{}/obs".format(ep)],
        )

        traj["rewards"] = dataset["data/{}/rewards".format(ep)][()]
        traj["dones"] = dataset["data/{}/dones".format(ep)][()]

        # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
        #            consistent as well
        grp_name = f"task_{task.name}_{ep}"
        ep_data_grp = data_grp.create_group(grp_name)
        # ep_data_grp.create_dataset("action", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
        # ep_data_grp.create_dataset("action", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset(f"action/cartesian_pose", data=np.array(traj["actions"])[:, :6])
        ep_data_grp.create_dataset(f"action/gripper_position", data=np.array(traj["actions"])[:, 6:])

        for k in traj["obs"]:
            if '_image' in k:
                ep_data_grp.create_dataset("observation/{}".format(k.replace('robot0_', '').replace('_image', '_rgb')), data=np.array(traj["obs"][k]))
            else:
                ep_data_grp.create_dataset("observation/{}".format(k), data=np.array(traj["obs"][k]))
        ep_data_grp.create_dataset("observation/{}".format("joint_states"), data=np.array(traj["obs"]["robot0_joint_pos"]))
        ep_data_grp.create_dataset("observation/{}".format("gripper_states"), data=np.array(traj["obs"]["robot0_gripper_qpos"]))
        ep_data_grp.create_dataset("observation/{}".format("gripper_position"), data=np.array(traj["obs"]["robot0_gripper_qpos"])[:,:1]) # only gripper position
        ep_data_grp.create_dataset("observation/{}".format("cartesian_pose"), data=np.array(traj["cartesian_pose"]))

        ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
        episode_length_json[grp_name] = traj["actions"].shape[0] # number of transitions in this episode
        total_samples += traj["actions"].shape[0]
        print("ep {}: wrote {} transitions to group {}".format(ind, ep_data_grp.attrs["num_samples"], ep))
        success = traj["success"]
        num_success += success

    print("Success rate: ", num_success / len(demos))
    # save the json file by the same name as the output file but with .json extension
    json_output_path = output_path[:-5] + '.json'
    with open(json_output_path, 'w') as f:
        json.dump(episode_length_json, f, cls=NpEncoder)

    json_output_path = output_path[:-5] + '_keys.json'
    keys = list(episode_length_json.keys())
    with open(json_output_path, 'w') as f:
        json.dump(keys, f, cls=NpEncoder)
    # global metadata
    data_grp.attrs["total"] = total_samples
    print("Wrote {} trajectories to {}".format(len(demos), output_path))

    f_out.close()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark_name",
        type=str,
        required=True,
        help="name of benchmark to use",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        required=True,
        help="index of task to process",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )

    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="path to config file",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview_image", "robot0_eye_in_hand_image"],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=224,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=224,
        help="(optional) width of image observations",
    )
    '''
    # change robot color to create domain mismatch
    parser.add_argument(
        "--change_robot_color",
        action='store_true',
        help="(optional) change robot color to create domain mismatch",
    )
    '''

    args = parser.parse_args()
    dataset_states_to_obs(args)
