"""
Based on robomimic/utils/train_utils.py

Contains utility functions for distilling MOMA project.
"""
import os
import time
import datetime
import shutil
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
from collections import OrderedDict, defaultdict


import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.file_utils as FileUtils

from robomimic.utils.dataset import SequenceDataset
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy

MODEL_TYPE_TO_DATAFILE_CONTENT = {
    "vae" : {
        "obs" : ["rgb"],
        "non_obs" : ["actions"],
    }
}

################################ Training Utils ################################

def get_exp_dir(config, auto_remove_exp_dir=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt 
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.
    
    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expanduser(config.train.output_dir)
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)
    if os.path.exists(base_output_dir):
        if not auto_remove_exp_dir:
            ans = input("WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(base_output_dir))
        else:
            ans = "y"
        if ans == "y":
            print("REMOVING")
            shutil.rmtree(base_output_dir)

    # only make model directory if model saving is enabled
    output_dir = None
    if config.experiment.save.enabled:
        output_dir = os.path.join(base_output_dir, time_str, "models")
        os.makedirs(output_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir)
    return log_dir, output_dir, video_dir

def load_data_for_training(config, obs_keys):
    """
    Data loading at the start of an algorithm.

    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

    Returns:
        train_dataset (SequenceDataset instance): train dataset object
        valid_dataset (SequenceDataset instance): valid dataset object (only if using validation)
    """

    # config can contain an attribute to filter on
    train_filter_by_attribute = config.train.hdf5_filter_key
    valid_filter_by_attribute = config.train.hdf5_validation_filter_key
    if valid_filter_by_attribute is not None:
        assert config.experiment.validate, "specified validation filter key {}, but config.experiment.validate is not set".format(valid_filter_by_attribute)

    # load the dataset into memory
    if config.experiment.validate:
        assert not config.train.hdf5_normalize_obs, "no support for observation normalization with validation data yet"
        assert (train_filter_by_attribute is not None) and (valid_filter_by_attribute is not None), \
            "did not specify filter keys corresponding to train and valid split in dataset" \
            " - please fill config.train.hdf5_filter_key and config.train.hdf5_validation_filter_key"
        train_demo_keys = FileUtils.get_demos_for_filter_key(
            hdf5_path=os.path.expanduser(config.train.data),
            filter_key=train_filter_by_attribute,
        )
        valid_demo_keys = FileUtils.get_demos_for_filter_key(
            hdf5_path=os.path.expanduser(config.train.data),
            filter_key=valid_filter_by_attribute,
        )
        assert set(train_demo_keys).isdisjoint(set(valid_demo_keys)), "training demonstrations overlap with " \
            "validation demonstrations!"
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute)
        valid_dataset = dataset_factory(config, obs_keys, filter_by_attribute=valid_filter_by_attribute)
    else:
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute)
        valid_dataset = None

    return train_dataset, valid_dataset

def dataset_factory(config, obs_keys, filter_by_attribute=None, dataset_path=None):
    """
    Create a SequenceDataset instance to pass to a torch DataLoader.

    Args:
        config (BaseConfig instance): config object

        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        dataset_path (str): if provided, the SequenceDataset instance should load
            data from this dataset path. Defaults to config.train.data.

    Returns:
        dataset (SequenceDataset instance): dataset object
    """
    if dataset_path is None:
        dataset_path = config.train.data

    ds_kwargs = dict(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=config.train.dataset_keys,
        load_next_obs=config.train.hdf5_load_next_obs, # whether to load next observations (s') from dataset
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=False,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        filter_by_attribute=filter_by_attribute
    )
    dataset = SequenceDataset(**ds_kwargs)

    return dataset

def run_rollout(
        policy, 
        env, 
        horizon,
        uncertainty_data_path,
        use_goals=False,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
    ):
    """
    Runs a rollout in an environment with the current network parameters.

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        env (EnvBase instance): environment to use for rollouts.

        horizon (int): maximum number of steps to roll the agent out for

        uncertainty_data_path (str): path to hdf5 dataset for uncertainty model

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        render (bool): if True, render the rollout to the screen

        video_writer (imageio Writer instance): if not None, use video writer object to append frames at 
            rate given by @video_skip

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

    Returns:
        results (dict): dictionary containing return, success rate, etc.

        expert_succeeded (bool): True if rollout was successful (i.e. expert was in control and finished the task successfully)        
    """

    def policy_needs_help():
        prob = 0.1
        if np.random.uniform() < prob:
            return True
        return False
    
    def get_skill():
        # dummy function that return skill action generator and skill name
        ac_generator = env.env.controller.grasp(env.env.objects["grasp_obj"]) #, track_obj=True) 
        skill_name = "grasp"
        return ac_generator, skill_name

    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)

    policy.start_episode()

    ob_dict = env.reset()
    ob_dict = env.reset()
    goal_dict = None

    results = {}
    video_count = 0  # video frame counter

    total_reward = 0.
    total_steps = 0
    success = { k: False for k in env.is_success() } # success metrics

    is_expert_in_control = False
    expert_succeeded = False

    student_traj = []
    
    n_addded_traj_il = 0
    n_added_traj_unc = 0

    try:
        for step_i in range(horizon):

            # NOTE : for now, once expert takes control, the expert finishes this episode without handing control back to student
            if not is_expert_in_control and policy_needs_help(): # TODO - implement this
                is_expert_in_control = True
            
            ######## Case: Expert is in Control ########
            # NOTE: this ignores rollout horizon and executes expert actions until success or failure
            if is_expert_in_control:

                ac_generator, skill_name = get_skill() # TODO - this is a dummy function. Implement function in environment that return correct skill action generator
                expert_results, expert_succeeded = env.execute_skill(ac_generator, skill_name, video_writer, video_skip=video_skip)

                # update results (student + expert)
                total_steps = step_i + expert_results["Horizon"]
                total_reward += expert_results["Return"]
                success["task"] = expert_succeeded

                # once the expert demonstration finish, end this rollout
                break
                    
            ######## Case: Student is in Control ########
            # else:
            #     # process observations (observations used in rollout must be processed)
            #     obs = env.get_observation(ob_dict, postprocess_for_eval=True)
            #     # get aciton from policy
            #     ac = policy(ob=obs, goal=goal_dict)
            #     # action from policy is normalized. unnormalize action before stepping
            #     ac = env.denormalize_action(ac)
            #     # step
            #     next_ob_dict, r, done, _ = env.step(ac)
            #     step_i += 1

            #     # render to screen - TODO 
            #     if render:
            #         env.render(mode="human")

            #     # compute reward
            #     total_reward += r

            #     cur_success_metrics = env.is_success()
            #     for k in success:
            #         success[k] = success[k] or cur_success_metrics[k]

            #     # visualization - # TODO: figure out offscreen rendering in og env (implement render function in moma_wrapper)
            #     if video_writer is not None:
            #         if video_count % video_skip == 0:
            #             video_img = env.render(mode="rgb_array", height=512, width=512)
            #             video_writer.append_data(video_img)

            #         video_count += 1
                
            #     # update ob_dict
            #     ob_dict = next_ob_dict

            #     # break if done
            #     if done or (terminate_on_success and success["task"]):
            #         break
            else:
                # process observations (observations used in rollout must be processed)
                obs = env.get_observation(ob_dict, postprocess_for_eval=True)
                # get aciton from policy
                ac = policy(ob=obs, goal=goal_dict)
                # action from policy is normalized. unnormalize action before stepping
                denormalized_ac = env.denormalize_action(ac)
                # step
                next_ob_dict, r, done, _ = env.step(denormalized_ac)
                step_i += 1

                # store student step
                step_data = {}
                step_data["obs"] = env.process_obs(ob_dict)
                step_data["action"] = ac
                step_data["reward"] = r
                step_data["done"] = done
                step_data["next_obs"] = env.process_obs(next_ob_dict)
                student_traj.append(step_data)

                # render to screen - TODO 
                if render:
                    env.render(mode="human")

                # compute reward
                total_reward += r

                cur_success_metrics = env.is_success()
                for k in success:
                    success[k] = success[k] or cur_success_metrics[k]

                # visualization - # TODO: figure out offscreen rendering in og env (implement render function in moma_wrapper)
                if video_writer is not None:
                    if video_count % video_skip == 0:
                        video_img = env.render(mode="rgb_array", height=512, width=512)
                        video_writer.append_data(video_img)

                    video_count += 1
                
                # update ob_dict
                ob_dict = next_ob_dict

                # break if done
                if done or (terminate_on_success and success["task"]):
                    break

        print(f"============== finished one rollout ({total_steps} steps) ==============")
        # if task succeeded, aggregate datasets
        if success["task"]:
            # uncertainty dataset
            skill_type, expert_traj = env.get_current_traj_history()[0]
            with h5py.File(uncertainty_data_path, "r+") as file:
                # TODO - get model type in a proper way
                # add expert component
                if len(expert_traj) > 0:
                    aggregate_uncertainty_detector_dataset(expert_traj, skill_type, file, "vae")
                    n_added_traj_unc += 1
                # add student component
                if len(student_traj) > 0:
                    aggregate_uncertainty_detector_dataset(student_traj, "none", file, "vae")
                    n_added_traj_unc += 1        
            # BC dataset can be aggregated using wrapper function
            env.flush_current_traj()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))
        # something went wrong. don't store trajectory

    results["Return"] = total_reward
    results["Horizon"] = total_steps + 1
    results["Success_Rate"] = float(success["task"])

    # log additional success metrics
    for k in success:
        if k != "task":
            results["{}_Success_Rate".format(k)] = float(success[k])

    return results, expert_succeeded, n_added_traj_unc

def rollout_with_stats(
        policy,
        envs,
        horizon,
        uncertainty_data_path,
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
    Variant of train_utils.rollout_with_stats: A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Agent actively queries expert when uncertain, and new expert trajectories are returned in addition to rollout_with_stats function.
    
    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        hdf5_path (str): path to hdf5 dataset to add expert trajectories to
        
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

        n_added_traj (int): number of expert trajectories added to dataset during rollout
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

    total_il_traj_added = 0 # number of expert trajectories added to dataset
    total_unc_traj_added = 0

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
            print(f"episode {ep_i+1} / {num_episodes}")
            rollout_timestamp = time.time()
            rollout_info, rollout_suuccess, n_added_traj_unc = run_rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                uncertainty_data_path=uncertainty_data_path,
                render=render,
                use_goals=use_goals,
                video_writer=env_video_writer,
                video_skip=video_skip,
                terminate_on_success=terminate_on_success,
            )
            total_unc_traj_added += n_added_traj_unc
            if rollout_suuccess:
                total_il_traj_added += 1
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

    return all_rollout_logs, video_paths, total_il_traj_added, total_unc_traj_added

def should_save_from_rollout_logs(
        all_rollout_logs,
        best_return,
        best_success_rate,
        epoch_ckpt_name,
        save_on_best_rollout_return,
        save_on_best_rollout_success_rate,
    ):
    """
    Helper function used during training to determine whether checkpoints and videos
    should be saved. It will modify input attributes appropriately (such as updating
    the best returns and success rates seen and modifying the epoch ckpt name), and
    returns a dict with the updated statistics.

    Args:
        all_rollout_logs (dict): dictionary of rollout results that should be consistent
            with the output of @rollout_with_stats

        best_return (dict): dictionary that stores the best average rollout return seen so far
            during training, for each environment

        best_success_rate (dict): dictionary that stores the best average success rate seen so far
            during training, for each environment

        epoch_ckpt_name (str): what to name the checkpoint file - this name might be modified
            by this function

        save_on_best_rollout_return (bool): if True, should save checkpoints that achieve a 
            new best rollout return

        save_on_best_rollout_success_rate (bool): if True, should save checkpoints that achieve a 
            new best rollout success rate

    Returns:
        save_info (dict): dictionary that contains updated input attributes @best_return,
            @best_success_rate, @epoch_ckpt_name, along with two additional attributes
            @should_save_ckpt (True if should save this checkpoint), and @ckpt_reason
            (string that contains the reason for saving the checkpoint)
    """
    should_save_ckpt = False
    ckpt_reason = None
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]

        if rollout_logs["Return"] > best_return[env_name]:
            best_return[env_name] = rollout_logs["Return"]
            if save_on_best_rollout_return:
                # save checkpoint if achieve new best return
                epoch_ckpt_name += "_{}_return_{}".format(env_name, best_return[env_name])
                should_save_ckpt = True
                ckpt_reason = "return"

        if rollout_logs["Success_Rate"] > best_success_rate[env_name]:
            best_success_rate[env_name] = rollout_logs["Success_Rate"]
            if save_on_best_rollout_success_rate:
                # save checkpoint if achieve new best success rate
                epoch_ckpt_name += "_{}_success_{}".format(env_name, best_success_rate[env_name])
                should_save_ckpt = True
                ckpt_reason = "success"

    # return the modified input attributes
    return dict(
        best_return=best_return,
        best_success_rate=best_success_rate,
        epoch_ckpt_name=epoch_ckpt_name,
        should_save_ckpt=should_save_ckpt,
        ckpt_reason=ckpt_reason,
    )

def save_model(model, config, env_meta, shape_meta, ckpt_path, obs_normalization_stats=None):
    """
    Save model to a torch pth file.

    Args:
        model (Algo instance): model to save

        config (BaseConfig instance): config to save

        env_meta (dict): env metadata for this training run

        shape_meta (dict): shape metdata for this training run

        ckpt_path (str): writes model checkpoint to this path

        obs_normalization_stats (dict): optionally pass a dictionary for observation
            normalization. This should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.
    """
    env_meta = deepcopy(env_meta)
    shape_meta = deepcopy(shape_meta)
    params = dict(
        model=model.serialize(),
        config=config.dump(),
        algo_name=config.algo_name,
        env_metadata=env_meta,
        shape_metadata=shape_meta,
    )
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        obs_normalization_stats = deepcopy(obs_normalization_stats)
        params["obs_normalization_stats"] = TensorUtils.to_list(obs_normalization_stats)
    torch.save(params, ckpt_path)
    print("save checkpoint to {}".format(ckpt_path))

def run_epoch(model, data_loader, epoch, validate=False, num_steps=None, obs_normalization_stats=None):
    """
    Run an epoch of training or validation.

    Args:
        model (Algo instance): model to train

        data_loader (DataLoader instance): data loader that will be used to serve batches of data
            to the model

        epoch (int): epoch number

        validate (bool): whether this is a training epoch or validation epoch. This tells the model
            whether to do gradient steps or purely do forward passes.

        num_steps (int): if provided, this epoch lasts for a fixed number of batches (gradient steps),
            otherwise the epoch is a complete pass through the training dataset

        obs_normalization_stats (dict or None): if provided, this should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.

    Returns:
        step_log_all (dict): dictionary of logged training metrics averaged across all batches
    """
    epoch_timestamp = time.time()
    if validate:
        model.set_eval()
    else:
        model.set_train()
    if num_steps is None:
        num_steps = len(data_loader)

    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])
    start_time = time.time()

    data_loader_iter = iter(data_loader)
    for _ in LogUtils.custom_tqdm(range(num_steps)):

        # load next batch from data loader
        try:
            t = time.time()
            batch = next(data_loader_iter)
        except StopIteration:
            # reset for next dataset pass
            data_loader_iter = iter(data_loader)
            t = time.time()
            batch = next(data_loader_iter)
        timing_stats["Data_Loading"].append(time.time() - t)

        # process batch for training
        t = time.time()
        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=obs_normalization_stats)
        timing_stats["Process_Batch"].append(time.time() - t)

        # forward and backward pass
        t = time.time()
        info = model.train_on_batch(input_batch, epoch, validate=validate)
        timing_stats["Train_Batch"].append(time.time() - t)

        # tensorboard logging
        t = time.time()
        step_log = model.log_info(info)
        step_log_all.append(step_log)
        timing_stats["Log_Info"].append(time.time() - t)

    # flatten and take the mean of the metrics
    step_log_dict = {}
    for i in range(len(step_log_all)):
        for k in step_log_all[i]:
            if k not in step_log_dict:
                step_log_dict[k] = []
            step_log_dict[k].append(step_log_all[i][k])
    step_log_all = dict((k, float(np.mean(v))) for k, v in step_log_dict.items())

    # add in timing stats
    for k in timing_stats:
        # sum across all training steps, and convert from seconds to minutes
        step_log_all["Time_{}".format(k)] = np.sum(timing_stats[k]) / 60.
    step_log_all["Time_Epoch"] = (time.time() - epoch_timestamp) / 60.

    return step_log_all

def is_every_n_steps(interval, current_step, skip_zero=False):
    """
    Convenient function to check whether current_step is at the interval. 
    Returns True if current_step % interval == 0 and asserts a few corner cases (e.g., interval <= 0)
    
    Args:
        interval (int): target interval
        current_step (int): current step
        skip_zero (bool): whether to skip 0 (return False at 0)

    Returns:
        is_at_interval (bool): whether current_step is at the interval
    """
    if interval is None:
        return False
    assert isinstance(interval, int) and interval > 0
    assert isinstance(current_step, int) and current_step >= 0
    if skip_zero and current_step == 0:
        return False
    return current_step % interval == 0


################################ Datafile and DataLoader Utils ################################

def initialize_dataloaders(config, shape_meta):
    """
    Returns training and validation data loaders.
    Modularized since MP-DAgger updates datafile during rollout and Dataloaders must be reinitialized

    Returns:
        train_loader (DataLoader instance): data loader for training dataset

        valid_loader (DataLoader instance): data loader for validation dataset

        obs_normalization_stats (dict or None): if provided, this should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.
    """
    # load training data
    trainset, validset = load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()
        raise Warning("normalize obs is not recommended")

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )
    else:
        valid_loader = None

    return train_loader, valid_loader, obs_normalization_stats

def create_uncertainty_detector_dataset(srcfile_path, model_type):
    """
    Create a new dataset for uncertainty detection model from a full dataset. The new dataset contains only data that is relevant to the specified uncertainty detector.
    
    Args:
        srcfile_path (str): path to full dataset
        model_type (str): type of uncertainty detector model. One of ["vae",...]

    Returns:
        new_datapath: path to new dataset
    """

    assert model_type in MODEL_TYPE_TO_DATAFILE_CONTENT.keys(), "model_type must be one of {}".format(MODEL_TYPE_TO_DATAFILE_CONTENT.keys())
    keys_to_add = MODEL_TYPE_TO_DATAFILE_CONTENT[model_type]

    srcfile = h5py.File(srcfile_path, "r")

    # if file exists at dstfile_path, delete it and create a new file
    data_dir = os.path.dirname(srcfile_path)
    dstfile_path = os.path.join(data_dir, f"{model_type}_" + os.path.basename(srcfile_path))
    if os.path.isfile(dstfile_path):
        os.remove(dstfile_path)
    dstfile = h5py.File(dstfile_path, "w")

    # copy over data universal to all uncertainty models
    data_grp = dstfile.create_group("data")
    mask_grp = dstfile.create_group("mask")
    data_grp.attrs["env_args"] = srcfile["data"].attrs["env_args"]
    data_grp.attrs["total"] = srcfile["data"].attrs["total"]
    for skill_type, grps in srcfile["mask"].items():
        mask_grp.create_dataset(skill_type, data=grps)

    for demo_name in srcfile["data"].keys():
        demo_grp = data_grp.create_group(demo_name)
        demo_grp.attrs["num_samples"] = srcfile["data"][demo_name].attrs["num_samples"]

        # copy over relevant observations
        if len(keys_to_add["obs"]) > 0:
            obs_grp = demo_grp.create_group("obs")
            next_obs_grp = demo_grp.create_group("next_obs")
            for mod in keys_to_add["obs"]:
                obs_grp.create_dataset(mod, data=srcfile["data"][demo_name]["obs"][mod])
                next_obs_grp.create_dataset(mod, data=srcfile["data"][demo_name]["next_obs"][mod])

        # copy over other relevant data (acitons, rewards, dones, etc.)
        for non_obs in keys_to_add["non_obs"]:
            demo_grp.create_dataset(non_obs, data=srcfile["data"][demo_name][non_obs])

    srcfile.close()
    dstfile.close()

    return dstfile_path

def aggregate_uncertainty_detector_dataset(traj_data, skill_type, hdf5_file, model_type):
    """
    Aggregates uncertainty detector dataset

    Args:
        traj_data (list) : single trajectory as a list of dictionaries
        hdf5_file (h5py.File) : hdf5 file to add the trajectory to
        model_type (str): type of uncertainty detector model. One of ["vae",...]
    """

    assert model_type in MODEL_TYPE_TO_DATAFILE_CONTENT.keys(), "model_type must be one of {}".format(MODEL_TYPE_TO_DATAFILE_CONTENT.keys())
    keys_to_add = MODEL_TYPE_TO_DATAFILE_CONTENT[model_type]

    n_demos = len(list(hdf5_file["data"].keys()))

    data_grp = hdf5_file.require_group("data")
    traj_grp = data_grp.create_group(f"demo_{n_demos}")
    traj_grp.attrs["num_samples"] = len(traj_data)

    obss = defaultdict(list)
    next_obss = defaultdict(list)
    actions = []
    rewards = []
    dones = []

    for step_data in traj_data:
        for mod, step_mod_data in step_data["obs"].items():
            if mod in keys_to_add["obs"]:
                obss[mod].append(step_mod_data)
        for mod, step_mod_data in step_data["next_obs"].items():
            if mod in keys_to_add["obs"]:
                next_obss[mod].append(step_mod_data)
        actions.append(step_data["action"])
        rewards.append(step_data["reward"])
        dones.append(step_data["done"])

    obs_grp = traj_grp.create_group("obs")
    for mod, traj_mod_data in obss.items():
        obs_grp.create_dataset(mod, data=np.stack(traj_mod_data, axis=0))
    next_obs_grp = traj_grp.create_group("next_obs")
    for mod, traj_mod_data in next_obss.items():
        next_obs_grp.create_dataset(mod, data=np.stack(traj_mod_data, axis=0))

    if "actions" in keys_to_add:
        traj_grp.create_dataset("actions", data=np.stack(actions, axis=0))
    if "rewards" in keys_to_add:
        traj_grp.create_dataset("rewards", data=np.stack(rewards, axis=0))
    if "dones" in keys_to_add:
        traj_grp.create_dataset("dones", data=np.stack(dones, axis=0))

    # update total step and skill mask in hdf5 file
    data_grp.attrs["total"] = len(traj_data)
    mask_grp = hdf5_file.require_group("mask")
    if skill_type in mask_grp.keys():
        old_data = [data.decode("utf-8") for data in mask_grp[skill_type][:]]
        del mask_grp[skill_type]
    new_data = old_data + [f"demo_{n_demos}"]
    mask_grp.create_dataset(skill_type, data=new_data)
