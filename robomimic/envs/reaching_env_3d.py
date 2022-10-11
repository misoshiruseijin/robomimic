"""
3D reaching environment for real Franka
"""
import sys
from turtle import pd
sys.path.insert(1, "/home/brainiacx-ws/robot-infra/robot_infra/gprs")
sys.path.insert(1, "/home/brainiacx-ws/robomimic/robomimic")

from gprs.franka_interface import FrankaInterface
from gprs.utils import YamlConfig
import gprs.utils.transform_utils as T

from robomimic.utils.myutils import normalize_rpy

import numpy as np
import time
import pdb
import math
import gym

class Franka3DReachingEnv(gym.Env):

    """
    maintain_eef_orn: if True, attempts to keep original end-effector orientation when using OSC_POSE controller
    random_init: if True, randomize end-effector starting position with each reset
    enable_gripper: if False, keeps gripper open regardless of input action
    terminate_on_success: if True, terminates episode when task succeeds
    """

    def __init__(
        self,
        env_name,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        postprocess_visual_obs=True,
        control_type="OSC_POSE",
        control_freq=20,
        maintain_eef_orn=True,
        enable_gripper=False,
        random_init=False,
        terminate_on_success=True,
        max_episode_steps=200,
    ):
        self.robot_interface = FrankaInterface(
            general_cfg_file="/home/brainiacx-ws/robot-infra/robot_infra/gprs/config/alice.yml",
            control_freq=control_freq,
            use_visualizer=False,
            )
        self.controller_cfg = YamlConfig("/home/brainiacx-ws/robot-infra/robot_infra/gprs/config/osc-controller.yml").as_easydict()
        self.control_type = control_type

        # clear state buffers
        self.robot_interface._state_buffer = []
        self.robot_interface._gripper_state_buffer = []
        self.robot_interface._history_actions = []

        # define reset (neutral) joint position
        self.reset_joint_positions = [0.03521453627525714, -0.5549872407089627, 0.30394139096193146, -2.8531993541665934, 0.20219325989524706, 2.3128883586192566, 0.9910985421153834]

        # workspace boundaries [lower limit, upper limit]
        self.workspace_x = [0.255, 0.76]
        self.workspace_y = [-0.33, 0.23]
        self.workspace_z = [0.135, 0.60]

        # target position [lower limit, upper limit]
        self.target_x = [0.52, 0.64]
        self.target_y = [-0.26, -0.11]
        self.target_z = [0.35, 0.50]

        self.random_init = random_init
        self.maintain_eef_orn = maintain_eef_orn
        self.terminate_on_success = terminate_on_success
        self.enable_gripper = enable_gripper

        self.max_episode_steps = max_episode_steps # number of steps per episode
        self.episode_steps = 0
        self.total_steps = 0
        self.episodes = 0 # number of episodes completed so far

        self.action_dimension = 7 # [dx, dy, dz, rotx, roty, rotz, gripper]

        self.last_eef_pos = None
        self.last_time = None

        self.initial_orn = np.array([0, 0, 0]) # orientation (RPY) to maintain when maintain_eef_orn is True
        self.last_orn = None

        self.prev_buffer_state = None
        self.reset()
    
    def step(self, action):

        action = np.array(action)
        # print("action: ", action)
        if action.shape[0] == 3: # correct action dimension
            action = np.concatenate([ np.array(action), np.array([0, 0, 0, -1]) ])

        # if given action moves end-effector outside 
        if not self._is_in_workspace(action):
            print("Action ignored: out of workspace limits")
            action[:-1] = 0 # set all delta position and orientation to zero (gripper can still be controlled)

        if not self.enable_gripper:
            action[-1] = -1

        # end-effector orientation correction - needs improvement
        if self.maintain_eef_orn:
            action[3:-1] = 0 
            if np.any(action[:3] != 0): # only rotate when user is giving input
                action = np.append(np.concatenate([ action[:3], self._get_orientation_correction() ]), action[-1])

        print("executing action: ", action)
        self.robot_interface.control(
            control_type=self.control_type,
            action=action,
            controller_cfg=self.controller_cfg,
        )
        
        observation = self.get_observation()
        
        print("state ", observation["eef_pos"])
        info = {}

        reward = self._get_reward()

        self.episode_steps += 1
        self.total_steps += 1

        done = self._is_done()
        if done: 
            self.episode_steps = 0

        return observation, reward, done, info

    def reset(self):
        action = self.reset_joint_positions + [-1.]

        print("-----------------Resetting-------------------")

        while True:
            if len(self.robot_interface._state_buffer) > 0:
                # print("Current joint positions: ", self.robot_interface._state_buffer[-1].q)
                # print(np.max(np.abs(np.array(self.robot_interface._state_buffer[-1].q) - np.array(self.reset_joint_positions))))
                if np.max(np.abs(np.array(self.robot_interface._state_buffer[-1].q) - np.array(self.reset_joint_positions))) < 1.5e-3:
                    break
            
            self.robot_interface.control(
                control_type="JOINT_POSITION",
                action=action,
                controller_cfg=self.controller_cfg
                )

        print("---------------Reset Complete------------------")

        if self.random_init:
            self._random_init()

        observation = self.get_observation()

        print("state after reset ", observation["eef_pos"])

        return observation

    def render(self):
        return None
    
    def get_observation(self):
        
        """
        Return state: dict
        Keys:
            eef_orn_mat (3x3 array) = end-effector orientation as rotation matrix in base frame
            eef_orn_rpy () = end-effector orientation as Euler angles (RPY)
            eef_pos (3d array) = Cartesian end-effector position in base frame
            eef_vel (3d array) = Cartesian end-effector velocity (computed via linear interpolation)
            gripper_width (float) = width of gripper
            q (7d array) = joint positions
            dq (7d array) = joint velocities
        """
        state = {}
        eef_pose = np.array(self.robot_interface._state_buffer[-1].O_T_EE)
        eef_pose = np.reshape(eef_pose, (4, 4)).T # eef_pose = [R | T]
        eef_orn_mat = eef_pose[:3, :3]
        state["eef_orn_mat"] = eef_orn_mat
        state["eef_pos"] = eef_pose[:3, -1]

        # interpolate to get end-effector cartesian velocity
        curr_time = time.time()
        if self.last_eef_pos is None: # computed for first time in reset
            state["eef_vel"] = np.zeros(3)
        else:
            state["eef_vel"] = (state["eef_pos"] - self.last_eef_pos) / (curr_time - self.last_time)
        
        self.last_eef_pos = state["eef_pos"]
        self.last_time = curr_time
        
        # maintaining orientation
        if self.initial_orn is None:
            self.initial_orn = T.mat2euler(eef_orn_mat)
            self.initial_orn = normalize_rpy(self.initial_orn)

        orn_rpy = T.mat2euler(eef_orn_mat)
        orn_rpy = normalize_rpy(orn_rpy)
        state["eef_orn_rpy"] = orn_rpy
        self.last_eef_orn = state["eef_orn_rpy"]
        
        state["gripper_width"] = np.array([self.robot_interface._gripper_state_buffer[-1].width])
        state["q"] = np.array(self.robot_interface._state_buffer[-1].q)
        state["dq"] = np.array(self.robot_interface._state_buffer[-1].dq)
        return state

    def _random_init(self, steps=15):
            print("......Taking random action")            
            x_action = np.random.uniform(-0.5, 0.5, 1)
            y_action = np.random.uniform(-0.5, 0.5, 1)
            z_action = np.random.uniform(-0.4, 0.4, 1)
            random_action = np.array([ x_action[0], y_action[0], z_action[0], 0, 0, 0, -1 ])

            for _ in range(steps):

                self.robot_interface.control(
                    control_type=self.control_type,
                    action=random_action,
                    controller_cfg=self.controller_cfg,
                )

            print("Finished random init")

            self.robot_interface.control(
                control_type=self.control_type,
                action=[0., 0., 0., 0., 0., 0., -1.],
                controller_cfg=self.controller_cfg,
            )

    def _get_reward(self):
        
        if self._is_success():
            print("------------Success--------------")
            return 10

        return 0

    def _is_success(self):
        """
        Returns:
            True if end-effector is in target area, False otherwise
        """
        current_pos = self.get_observation()["eef_pos"]
        in_x = current_pos[0] > self.target_x[0] and current_pos[0] < self.target_x[1]
        in_y = current_pos[1] > self.target_y[0] and current_pos[1] < self.target_y[1]
        in_z = current_pos[2] > self.target_z[0] and current_pos[2] < self.target_z[1]

        success = in_x and in_y and in_z
        return success

    def _is_done(self):
        
        if self.terminate_on_success and self._is_success():
            return True

        if self.episode_steps >= self.max_episode_steps:
            return True

        return False

    def _is_in_workspace(self, action):
        """
        Returns:
            True if action moves end-effector out of the workspace limits,
            False otherwise
        """

        control_scale = 0.05 # action is scaled by this value before used in controller (see franka_interface.py)
        next_pos = np.array(self.get_observation()["eef_pos"]) + control_scale * np.array(action[:3])
        in_x = next_pos[0] > self.workspace_x[0] and next_pos[0] < self.workspace_x[1]
        in_y = next_pos[1] > self.workspace_y[0] and next_pos[1] < self.workspace_y[1]
        in_z = next_pos[2] > self.workspace_z[0] and next_pos[2] < self.workspace_z[1]
        
        print(f"in xyz {in_x} {in_y} {in_z}")
        return in_x and in_y and in_z
    
    def _get_orientation_correction(self):
        """
        Returns:
            Rotational component of action (RPY) to maintain initial orientation
        """
        gain = 2
        action_rotational = self.get_observation()["eef_orn_rpy"] - self.initial_orn

        return np.clip(-gain * action_rotational, -0.35, 0.35)
   







