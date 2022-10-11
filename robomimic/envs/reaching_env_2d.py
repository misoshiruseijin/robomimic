"""
3D reaching environment for real Franka
"""
import sys
from turtle import pd
sys.path.insert(1, "/home/brainiacx-ws/robot-infra/robot_infra/gprs")
# sys.path.insert(1, "/home/brainiacx-ws/robomimic/robomimic")

from gprs.franka_interface import FrankaInterface
from gprs.utils import YamlConfig
import gprs.utils.transform_utils as T

from robomimic.utils.myutils import normalize_rpy

import numpy as np
import time
import pdb
import math
import gym

class Franka2DReachingEnv(gym.Env):

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
        # self.reset_joint_positions = [0.03521453627525714, -0.5549872407089627, 0.30394139096193146, -2.8531993541665934, 0.20219325989524706, 2.3128883586192566, 0.9910985421153834]
        self.reset_joint_positions = [0.0, -0.5549872407089627, 0.0, -2.8531993541665934, 0.0, 2.3128883586192566, 0.90]

        # workspace boundaries [lower limit, upper limit]
        self.workspace_x = [0.255, 0.76]
        self.workspace_y = [-0.33, 0.23]
        self.workspace_z = [0.135, 0.60]

        # target position [lower limit, upper limit]
        # self.target_x = [0.52, 0.64]
        # self.target_y = [-0.26, -0.11]
        initial_eef_pos = np.array([0.34, 0])
        self.target_x = np.array([0.2, 0.3]) + initial_eef_pos[0] # center of target is 0.25m away from home position
        self.target_y = np.array([-0.125, 0.125]) + initial_eef_pos[1]
        self.target_center = np.array([np.mean(self.target_x), np.mean(self.target_y)])

        self.random_init = random_init
        self.maintain_eef_orn = maintain_eef_orn
        self.terminate_on_success = terminate_on_success

        self.max_episode_steps = max_episode_steps # number of steps per episode
        self.episode_steps = 0
        self.total_steps = 0
        self.episodes = 0 # number of episodes completed so far

        self.action_dimension = 7 # [dx, dy, dz, rotx, roty, rotz, gripper]

        self.control_freq = control_freq

        self.last_eef_pos = None
        self.last_time = None

        self.initial_orn = np.array([0, 0, 0]) # orientation (RPY) to maintain when maintain_eef_orn is True
        self.last_orn = None

        self.initial_z = 0.235

        self.prev_buffer_state = None

        self.in_target = False

        # initial eef position: 0.34, 0, 0.24

        self.reset()
    
    def step(self, action):

        if action.shape[0] == 2: # correct action dimension
            action = np.concatenate([ np.array(action), np.array([0, 0, 0, 0, -1]) ])
            # action = np.concatenate([ np.array(action), -1 ]) # for position control

        # print("action (raw): ", action)

        # if given action moves end-effector outside 
        if not self._is_in_workspace(action):
            print("----Action ignored: out of workspace limits----")
            action[:-1] = 0 # set all delta position and orientation to zero (gripper can still be controlled)

        # end-effector orientation and z position correction
        if np.any(action[:3] != 0): # do not move when input action is zero
            if self.maintain_eef_orn:
                action = np.append(np.concatenate([ action[:3], self._get_orientation_correction() ]), action[-1])
                # action = np.append(np.concatenate([ action[:3], np.zeros(3) ]), action[-1]) # no orientation correction
                # action = np.concatenate((action[:3], action[-1:]), axis=0) # for position control

            action[2] = self._get_height_correction() # height correction
        print("executed action: ", action[:2])

        self.robot_interface.control(
            control_type=self.control_type,
            action=action,
            controller_cfg=self.controller_cfg,
        )

        time.sleep(0.1)
        observation = self.get_observation()
        self.in_target = self._is_success()

        reward = self._get_reward()
        
        info = {}   
        info["in_target"] = self.in_target
        
        self.episode_steps += 1
        self.total_steps += 1
        
        done = self._is_done()
        if done: 
            self.episode_steps = 0

        print("state after step ", observation["eef_pos"])
        print("distance to target center ", np.linalg.norm(observation["eef_pos"][:2] - self.target_center))
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
            time.sleep(1)
            self._random_init()

        observation = self.get_observation()

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
        new_state = self.robot_interface._state_buffer[-1]
        new_gripper_state = self.robot_interface._gripper_state_buffer[-1]
        
        eef_pose = np.array(new_state.O_T_EE)
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
        
        state["gripper_width"] = np.array([new_gripper_state.width])
        state["q"] = np.array(new_state.q)
        state["dq"] = np.array(new_state.dq)
        return state

    def _random_init(self, steps=15, xrange=(0., 0.5), yrange=(-0.5, 0.5)):
            
            print("......Taking random action")       
            x_action = np.random.uniform(xrange[0], xrange[1], 1)
            y_action = np.random.uniform(yrange[0], yrange[1], 1)
            random_action = np.array([ x_action[0], y_action[0], 0, 0, 0, 0, -1 ])
            print("taking random action: ", random_action)
            print("random init steps: ", steps)
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
        
        if self.in_target:
            print("------------Success--------------")
            return 100

        return 0

    def _is_success(self):
        """
        Returns:
            True if end-effector is in target area, False otherwise
        """
        current_pos = self.get_observation()["eef_pos"]
        in_x = current_pos[0] > self.target_x[0] and current_pos[0] < self.target_x[1]
        in_y = current_pos[1] > self.target_y[0] and current_pos[1] < self.target_y[1]

        success = in_x and in_y
        print("in x, in y: ", in_x, in_y)
        return success

    def _is_done(self):
        
        if self.terminate_on_success and self.in_target:
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

        # print(f"in xyz {in_x} {in_y} {in_z}")
        return in_x and in_y and in_z
    
    def _get_orientation_correction(self):
        """
        Returns:
            Rotational component of action (RPY) to maintain initial orientation
        """
        gain = 2
        action_rotational = self.get_observation()["eef_orn_rpy"] - self.initial_orn

        return np.clip(-gain * action_rotational, -0.35, 0.35)
   
    def _get_height_correction(self):
        """
        Returns:
            (int) z component of action to maintain initial height
        """
        gain = 5
        action_z = self.get_observation()["eef_pos"][2] - self.initial_z
        return -gain * action_z





