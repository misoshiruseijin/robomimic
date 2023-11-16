"""
Customized for distilling-moma project

The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

import argparse
import json
import numpy as np
import time
import os
import shutil
import psutil
import sys
import socket
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.distilling_moma_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings

def rollout_and_aggregate(
    config, 
    model,
    envs,
    video_dir,
    epoch,
    data_logger,
    obs_normalization_stats,
    train_loader,
    valid_loader,
    unc_data_path,
    best_return,
    best_success_rate,
    unc_model,
    unc_device,
    query_method,
):
    # do rollouts at fixed rate or if it's time to save a new ckpt
    video_paths = None

    # wrap model as a RolloutPolicy to prepare for rollouts
    rollout_model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)

    num_episodes = config.experiment.rollout.n

    # Datafile gets aggregated here
    all_rollout_logs, video_paths, n_added_il_traj, n_added_unc_traj = TrainUtils.rollout_with_stats(
        policy=rollout_model,
        envs=envs,
        horizon=config.experiment.rollout.horizon,
        use_goals=config.use_goals,
        num_episodes=num_episodes,
        render=False,
        video_dir=video_dir if config.experiment.render_video else None, # TODO rendering is not supported yet - implement it in moma_wrapper
        epoch=epoch,
        video_skip=config.experiment.get("video_skip", 5),
        terminate_on_success=config.experiment.rollout.terminate_on_success,
        uncertainty_data_path=unc_data_path,
        unc_model=unc_model,
        unc_device=unc_device,
        query_method=query_method,
    )

    # summarize results from rollouts to tensorboard and terminal
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]
        for k, v in rollout_logs.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
            else:
                data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)
        print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
        print('Env: {}'.format(env_name))
        print(json.dumps(rollout_logs, sort_keys=True, indent=4))
    
    epoch_ckpt_name = "model_epoch_{}".format(epoch)

    # checkpoint and video saving logic
    updated_stats = TrainUtils.should_save_from_rollout_logs(
        all_rollout_logs=all_rollout_logs,
        best_return=best_return,
        best_success_rate=best_success_rate,
        epoch_ckpt_name=epoch_ckpt_name,
        save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
        save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
    )
    best_return = updated_stats["best_return"]
    best_success_rate = updated_stats["best_success_rate"]
    epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
    should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) 
    if updated_stats["ckpt_reason"] is not None:
        ckpt_reason = updated_stats["ckpt_reason"]

    # Only keep saved videos if the ckpt should be saved (but not because of validation score)
    should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
    if video_paths is not None and not should_save_video:
        for env_name in video_paths:
            os.remove(video_paths[env_name])
    
    return n_added_il_traj, n_added_unc_traj, best_return, best_success_rate

def train_loop(
    config,
    model, data_logger, train_loader, valid_loader,
    obs_normalization_stats,
    env_meta, shape_meta, ckpt_dir,
    best_valid_loss, last_ckpt_time, epochs_so_far,
):
    """
    Main training loop

    Args:
        config (dict): config dictionary
        model (Algo): model to train
        data_logger (DataLogger): data logger for logging training stats
        train_infos (dict): dictionary containing `best_valid_loss`, `best_return`, `best_success_rate`, and `last_ckpt_time`
        epochs_so_far (int): total number of epochs this model has been trained for
    """

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(epochs_so_far + 1, epochs_so_far + config.train.num_epochs_per_loop + 1): # epoch numbers start at 1
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and \
                (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
            epoch_list_check = (epoch in config.experiment.save.epochs)
            should_save_ckpt = (time_check or epoch_check or epoch_list_check)
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps)
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)

            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
            )

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # terminate logging
    # data_logger.close()

    return best_valid_loss, last_ckpt_time, epoch

def prepare_for_training(config, log_dir, shape_meta, device):
    """
    Creates data logger, model, saves config to log-dir, and set up dataloaders
    """
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )

    model = create_model_from_config(config, shape_meta, device)

    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    train_loader, valid_loader, obs_normalization_stats = TrainUtils.initialize_dataloaders(config=config, shape_meta=shape_meta)
    
    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")
    
    return data_logger, model, train_loader, valid_loader, obs_normalization_stats

def create_model_from_config(config, shape_meta, device):
    """
    Creates model from config
    """
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    return model

def train_mpdagger(il_config, unc_config, il_device, unc_device, il_model_ckpt=None, unc_model_ckpt=None, from_scratch_every_iter=False):
    # TODO - add option to load models from ckpt
    """
    Train a motion planner dagger with uncertainty estimation 
    """

    # first set seeds
    np.random.seed(il_config.train.seed)
    torch.manual_seed(il_config.train.seed)

    torch.set_num_threads(2)

    print("\n============= New Training Run with Config =============")
    print(il_config)
    print("")

    # get logging directories
    il_log_dir, il_ckpt_dir, il_video_dir = TrainUtils.get_exp_dir(il_config) 
    unc_log_dir, unc_ckpt_dir, unc_video_dir = TrainUtils.get_exp_dir(unc_config)

    if il_config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(il_log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(il_config)


    """
    DATA PREPARATION
        - confirm that dataset exists
        - create copy of dataset to aggregate (for IL policy)
        - create new dataset for uncertainty estimation model by copying relevant data    
    """
    # make sure IL dataset exists
    dataset_path = os.path.expanduser(il_config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))
    
    # Since training aggregates the datafile, make a copy of dataset and aggreate on the copy to avoid overwriting the original file
    # print("WARINING: LINE TO MAKE DATASET COPY IS COMMENTED OUT!")
    print("Making a copy of the dataset. This may take a while...")
    original_dataset_path = dataset_path
    student_aggr_dataset_path = dataset_path.split(".")[0] + "_aggr.hdf5"
    shutil.copy2(src=original_dataset_path, dst=student_aggr_dataset_path) # TODO - uncomment this
    print(f"Done making a copy of the dataset. Aggregated dataset will be saved at {student_aggr_dataset_path}")
    
    # overwrite config with new dataset path
    il_config.train.data = student_aggr_dataset_path
    
    # create new dataset for uncertainty estimation model
    # TODO - get model type in a proper way
    unc_dataset_path = TrainUtils.create_uncertainty_detector_dataset(dataset_path, "vae")
    unc_config.train.data = unc_dataset_path

    
    """
    ENVIRONMENT AND METADATA PREPARATION
        - load env metadata from IL dataset
        - load shape metadata for each dataset
        - create environment from IL config
        - initialize datalogger, dataloader, and model for IL and uncertainty estimation model
    """
    # load env metadata from IL dataset
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=il_config.train.data)
    
    # load shape metadata for each dataset
    il_shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=il_config.train.data,
        all_obs_keys=il_config.all_obs_keys,
        verbose=True
    )
    unc_shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=unc_config.train.data,
        all_obs_keys=unc_config.all_obs_keys,
        verbose=True
    )

    if il_config.experiment.env is not None:
        env_meta["env_name"] = il_config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment from IL config
    envs = OrderedDict()
    if il_config.experiment.rollout.enabled:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]
        if il_config.experiment.additional_envs is not None:
            for name in il_config.experiment.additional_envs:
                env_names.append(name)

        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False, 
                render_offscreen=il_config.experiment.render_video,
                use_image_obs=il_shape_meta["use_images"], 
            )
            env = EnvUtils.wrap_env_from_config(env, config=il_config, env_meta=env_meta, dataset_path=student_aggr_dataset_path) # apply environment warpper, if applicable
            envs[env.name] = env
            print(envs[env.name])

    print("")

    il_data_logger, il_model, il_train_loader, il_valid_loader, il_obs_normalization_stats = \
        prepare_for_training(il_config, il_log_dir, il_shape_meta, il_device, )
    
    unc_data_logger, unc_model, unc_train_loader, unc_valid_loader, unc_obs_normalization_stats = \
        prepare_for_training(unc_config, unc_log_dir, unc_shape_meta, unc_device, )

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")


    """
    MAIN TRAINING LOOP
        - train uncertainty estimation model
        - train IL model
        - rollout policy and aggregate datasets
        - repeat
    """
    # main training loop
    il_best_valid_loss = None
    il_best_return = {k: -np.inf for k in envs} if il_config.experiment.rollout.enabled else None
    il_best_success_rate = {k: -1. for k in envs} if il_config.experiment.rollout.enabled else None
    il_last_ckpt_time = time.time()
    unc_best_valid_loss = None
    unc_best_return = {k: -np.inf for k in envs} if il_config.experiment.rollout.enabled else None
    unc_best_success_rate = {k: -1. for k in envs} if il_config.experiment.rollout.enabled else None
    unc_last_ckpt_time = time.time()

    # number of epochs each model has been trained so far
    il_epochs_so_far = 0
    unc_epochs_so_far = 0

    done_training = (il_epochs_so_far >= il_config.train.num_epochs) and (unc_epochs_so_far >= unc_config.train.num_epochs)
    phases = ["unc", "il"]
    training_phase = 0 # start with training uncertainty estimation model
    # should_train_unc = True

    while not done_training:
        phase = phases[training_phase]
        if phase == "unc": # and should_train_unc:
            print("TRAINING UNCERTAINTY MODEL")
            # train the uncertainty estimation model
            unc_best_valid_loss, unc_last_ckpt_time, unc_epochs_so_far = train_loop(
                config=unc_config,
                model=unc_model,
                data_logger=unc_data_logger,
                train_loader=unc_train_loader,
                valid_loader=unc_valid_loader,
                obs_normalization_stats=unc_obs_normalization_stats,
                env_meta=env_meta,
                shape_meta=unc_shape_meta,
                ckpt_dir=unc_ckpt_dir,
                best_valid_loss=unc_best_valid_loss,
                last_ckpt_time=unc_last_ckpt_time,
                epochs_so_far=unc_epochs_so_far,
            )

        elif phase == "il":
            print("TRAINING IL MODEL")
            # train IL model
            il_best_valid_loss, il_last_ckpt_time, il_epochs_so_far = train_loop(
                config=il_config,
                model=il_model,
                data_logger=il_data_logger,
                train_loader=il_train_loader,
                valid_loader=il_valid_loader,
                obs_normalization_stats=il_obs_normalization_stats,
                env_meta=env_meta,
                shape_meta=il_shape_meta,
                ckpt_dir=il_ckpt_dir,
                best_valid_loss=il_best_valid_loss,
                last_ckpt_time=il_last_ckpt_time,
                epochs_so_far=il_epochs_so_far,
            )

            # Close hdf5 file before rollouts
            il_train_loader.dataset.close_and_delete_hdf5_handle()
            if il_valid_loader is not None:
                il_valid_loader.dataset.close_and_delete_hdf5_handle()
            unc_train_loader.dataset.close_and_delete_hdf5_handle()
            if unc_valid_loader is not None:
                unc_valid_loader.dataset.close_and_delete_hdf5_handle()

            # rollout policy
            n_added_il_traj, n_added_unc_traj, il_best_return, il_best_success_rate = rollout_and_aggregate(
                config=il_config,
                model=il_model,
                envs=envs,
                video_dir=il_video_dir,
                epoch=il_epochs_so_far,
                data_logger=il_data_logger,
                obs_normalization_stats=il_obs_normalization_stats,
                train_loader=il_train_loader,
                valid_loader=il_valid_loader,
                unc_data_path=unc_dataset_path,
                best_return=il_best_return,
                best_success_rate=il_best_success_rate,
                unc_model=unc_model,
                unc_device=unc_device,
                query_method="vae", # TODO - get this from config?
            )

            print(f"Added {n_added_il_traj} IL traj and {n_added_unc_traj} uncertainty traj to dataset")
            # if trajectories were added to dataset, reinitialize Dataset and DataLoader using updated hdf5 file
            if n_added_il_traj > 0:
                print(f"\n Added {n_added_il_traj} expert trajectories to IL dataset. Reinitializing dataloaders with updated datafile")
                il_train_loader, il_valid_loader, il_obs_normalization_stats = \
                    TrainUtils.initialize_dataloaders(config=il_config, shape_meta=il_shape_meta)
            if n_added_unc_traj > 0:
                print(f"\n Added {n_added_unc_traj} successful trajectories to uncertainty estimation dataset. Reinitializing dataloaders with updated datafile")
                unc_train_loader, unc_valid_loader, unc_obs_normalization_stats = \
                    TrainUtils.initialize_dataloaders(config=unc_config, shape_meta=unc_shape_meta)
                
            # should_train_unc = n_added_unc_traj > 0

            # Create new model if training models from scratch in the next loop
            if from_scratch_every_iter:
                il_model = create_model_from_config(il_config, il_shape_meta, il_device)
                unc_model = create_model_from_config(unc_config, unc_shape_meta, unc_device)
    
        done_training = (il_epochs_so_far >= il_config.train.num_epochs)\
            and (unc_epochs_so_far >= unc_config.train.num_epochs)
        training_phase = (training_phase + 1) % len(phases)

    # terminate logging
    il_data_logger.close()
    unc_data_logger.close()

def main(args):


    # if args.config is not None:
    #     ext_cfg = json.load(open(args.config, 'r'))
    #     config = config_factory(ext_cfg["algo_name"])
    #     # update config with external json - this will throw errors if
    #     # the external config has keys not present in the base algo config
    #     with config.values_unlocked():
    #         config.update(ext_cfg)
    # else:
    #     config = config_factory(args.algo)

    # if args.dataset is not None:
    #     config.train.data = args.dataset

    # if args.name is not None:
    #     config.experiment.name = args.name


    # get torch device
    # device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # # maybe modify config for debugging purposes
    # if args.debug:
    #     # shrink length of training to test whether this run is likely to crash
    #     config.unlock()
    #     config.lock_keys()

    #     # train and validate (if enabled) for 3 gradient steps, for 2 epochs
    #     config.experiment.epoch_every_n_steps = 3
    #     config.experiment.validation_epoch_every_n_steps = 3
    #     config.train.num_epochs = 2

    #     # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
    #     config.experiment.rollout.rate = 1
    #     config.experiment.rollout.n = 2
    #     config.experiment.rollout.horizon = 10

    #     # send output to a temporary directory
    #     config.train.output_dir = "/tmp/tmp_trained_models"

    # # lock config to prevent further modifications and ensure missing keys raise errors
    # config.lock()

    # # catch error during training and print it
    # res_str = "finished run successfully!"
    # try:
    #     train(config, device=device)
    # except Exception as e:
    #     res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    # print(res_str)

    # TODO - overwrite dataset path in config with args.dataset
    print("root directory", os.getcwd())
    ext_il_cfg = json.load(open(args.il_config, 'r'))
    ext_unc_cfg = json.load(open(args.unc_config, 'r'))

    if args.debug:
        # do 2 rollouts every 1 epoch
        ext_il_cfg["experiment"]["rollout"]["n"] = 2
        ext_il_cfg["experiment"]["rollout"]["rate"] = 1
        ext_il_cfg["experiment"]["rollout"]["horizon"] = 100
        ext_il_cfg["train"]["num_epochs_per_loop"] = 1
        ext_unc_cfg["train"]["num_epochs_per_loop"] = 1

    il_config = config_factory(ext_il_cfg["algo_name"])
    with il_config.values_unlocked():
        il_config.update(ext_il_cfg)

    unc_config = config_factory(ext_unc_cfg["algo_name"])

    with unc_config.values_unlocked():
        unc_config.update(ext_unc_cfg)

    # get torch device
    il_device = TorchUtils.get_torch_device(try_to_use_cuda=il_config.train.cuda)
    unc_device = TorchUtils.get_torch_device(try_to_use_cuda=unc_config.train.cuda)

    train_mpdagger(
        il_config=il_config, unc_config=unc_config,
        il_device=il_device, unc_device=unc_device,
        from_scratch_every_iter=args.scratch,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file for IL policy
    parser.add_argument(
        "--il-config",
        type=str,
        default="/distilling-moma/distilling_moma/experiment_configs/test_dagger_debug.json",
        help="path to a config json that will be used to override the default settings",
    )

    # External config file for uncertainty estimation model
    parser.add_argument(
        "--unc-config",
        type=str,
        # default="/distilling-moma/distilling_moma/experiment_configs/test_vae_debug.json",
        default="/distilling-moma/distilling_moma/experiment_configs/test_cvae_debug.json",
        help="path to a config json that will be used to override the default settings",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default="/viscam/u/ayanoh/distilling-moma/data/demo_100_traj_camera_v2.hdf5",
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    parser.add_argument(
        "--scratch",
        action='store_true',
        help="set this flag to retrain a model from scratch every time data aggregation happens"
    )

    args = parser.parse_args()
    main(args)

