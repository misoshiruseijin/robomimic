import numpy as np
import h5py
import pdb

def normalize_angle_positive(angles):
    """
    normalize angle in rad to range [0, 2pi]
    """
    pi2 = 2 * np.pi
    return np.fmod( np.fmod(angles, pi2) + pi2, pi2)

def normalize_rpy(angles):
    pi2 = 2 * np.pi
    result = angles.copy()
    result[0] = np.fmod( np.fmod(result[0], pi2) + pi2, pi2) - np.pi
    # angles[0] -= np.pi
    # angles[1:] -= 2 * np.pi
    return result

def add_dummy_states_to_dataset(dataset):
    """
    Add dummy state values to non-robosuite datasets for robomimic compatibility
    """
    dummy_state = np.zeros(1)
    f = h5py.File(dataset, 'r+')
    pdb.set_trace()
    for demo in f["data"].keys():
        if "states" in f[f"data/{demo}"].keys():
            # if states dataset already exist, remove it
            del f[f"data/{demo}/states"]

        f[f"data/{demo}"].create_dataset("states", data=dummy_state)

        # f[f"data/{demo}"].create_dataset("states", data=dummy_state)
    f.close()

if __name__ == "__main__":
    add_dummy_states_to_dataset(dataset="/home/brainiacx-ws/robomimic/franka_dagger_trained_models/aggr_dataset.hdf5")