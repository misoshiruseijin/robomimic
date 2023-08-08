import robomimic.utils.file_utils as FileUtils


dataset_path = "/home/ayanoh/robomimic/dagger_test_expert_models/bctest_10traj/bctraj_10train_1valid.hdf5"

env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

breakpoint()