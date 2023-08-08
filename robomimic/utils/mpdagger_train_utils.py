import os
import time
import datetime
import shutil
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.file_utils as FileUtils

from robomimic.utils.dataset import SequenceDataset
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy

def rollout_with_stats(
        policy,
        envs,
        horizon,
        use_goals=False,
        num_episodes=None,
        render=False,
        video_dir=None,
        video_path=None,
        epoch=None,
        video_skip=5,
        terminate_on_success=False,
        verbose=False,
    ):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        num_episodes (int): number of rollout episodes per environment

        render (bool): if True, render the rollout to the screen

        video_dir (str): if not None, dump rollout videos to this directory (one per environment)

        video_path (str): if not None, dump a single rollout video for all environments

        epoch (int): epoch number (used for video naming)

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        verbose (bool): if True, print results of each rollout
    
    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...) 
            averaged across all rollouts 

        video_paths (dict): path to rollout videos for each environment
    """
    assert isinstance(policy, RolloutPolicy)

    all_rollout_logs = OrderedDict()

    # handle paths and create writers for video writing
    assert (video_path is None) or (video_dir is None), "rollout_with_stats: can't specify both video path and dir"
    write_video = (video_path is not None) or (video_dir is not None)
    video_paths = OrderedDict()
    video_writers = OrderedDict()
    if video_path is not None:
        # a single video is written for all envs
        video_paths = { k : video_path for k in envs }
        video_writer = imageio.get_writer(video_path, fps=20)
        video_writers = { k : video_writer for k in envs }
    if video_dir is not None:
        # video is written per env
        video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4" 
        video_paths = { k : os.path.join(video_dir, "{}{}".format(k, video_str)) for k in envs }
        video_writers = { k : imageio.get_writer(video_paths[k], fps=20) for k in envs }

    for env_name, env in envs.items():
        env_video_writer = None
        if write_video:
            print("video writes to " + video_paths[env_name])
            env_video_writer = video_writers[env_name]

        print("rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
            env.name, horizon, use_goals, num_episodes,
        ))
        rollout_logs = []
        iterator = range(num_episodes)
        if not verbose:
            iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

        num_success = 0
        for ep_i in iterator:
            rollout_timestamp = time.time()
            rollout_info = run_rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                render=render,
                use_goals=use_goals,
                video_writer=env_video_writer,
                video_skip=video_skip,
                terminate_on_success=terminate_on_success,
            )
            rollout_info["time"] = time.time() - rollout_timestamp
            rollout_logs.append(rollout_info)
            num_success += rollout_info["Success_Rate"]
            if verbose:
                print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
                print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
        rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
        all_rollout_logs[env_name] = rollout_logs_mean

    if video_path is not None:
        # close video writer that was used for all envs
        video_writer.close()

    return all_rollout_logs, video_paths

