import yaml
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
# from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.usd_utils import RigidContactAPI
from omnigibson.object_states import OnTop
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives

import time

class MOMAEnv:
    """
    Wrapper for omniverse base Environment. For use with distilling-MOMA

    Args:
        config_filename (str): path to config file
        max_steps (int): maximum number of steps before termination
        random_init_base_pose (None or list): if None, initialize robot base at default pose or pose set by initial_robot_pose.
            if list, initialize robot at random pose within range specified by list ([x_low, x_high], [y_low, y_high], [yaw_low, yaw_high])
        initial_robot_pose (list): default initial base pose [x, y, yaw]
        raise_hand (bool): if True, raise hand at start of episode and set the tuck pose to raised hand pose
    """
    
    def __init__(
        self,
        name,
        config_filename,
        max_steps=1000,
        max_collisions=500, # TODO - implement this if needed
        random_init_base_pose=None, # randomly initialize robot base pose
        random_init_robot_pose=False, # randomly initialize robot base and arm poses - TODO
        initial_robot_pose=[0.0, 0.0, 0.0], # defualt initial pose [x, y, yaw]
        raise_hand=True,
        fix_arm_during_nav=True, # whether to overwrite main arm action during navigation to maintain default pose
    ):
        # Load the config
        self.config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        # Randomization settings
        self.random_init_base_pose = random_init_base_pose
        self.random_init_robot_pose = random_init_robot_pose

        # Load the environment
        self.env = og.Environment(configs=self.config)
        self.scene = self.env.scene
        self.robots = self.env.robots
        self.robot = self.env.robots[0]
        self.robot_model = self.robot.model_name

        # other settings
        self.name = name
        self.raise_hand = raise_hand
        self.initial_robot_pose = initial_robot_pose
        self.fix_arm_during_nav = fix_arm_during_nav
        if self.fix_arm_during_nav:
            pass # TODO - make sure controller is joint position (absolute) controller

        # initialize primitive skill controller
        self.controller = StarterSemanticActionPrimitives(
            task=None,
            scene=self.scene,
            robot=self.robot,
        )

        self.max_steps = max_steps
        self.max_collisions = max_collisions

        # Allow user to move camera more easily
        og.sim.enable_viewer_camera_teleoperation()

        self.objects = {}
        for obj in self.scene.objects:
            self.objects[obj.name] = obj
        
        # set robot to initial configuration
        self._initialize_robot_pose()


        self.steps = 0

    def _initialize_robot_pose(self):

        # set initial joint positions first before setting base pose
        # if not self.random_init_robot_pose:
        #     # self.tuck_arm()
        self.robot.set_joint_positions(self.default_joint_pos)
        
        # take a few steps
        self.step_sim(10)

        # set base pose
        if self.random_init_base_pose is not None:
            x_lim = self.random_init_base_pose[0]
            y_lim = self.random_init_base_pose[1]
            yaw_lim = self.random_init_base_pose[2]
            x = np.random.uniform(low=x_lim[0], high=x_lim[1], size=1)[0]
            y = np.random.uniform(low=y_lim[0], high=y_lim[1], size=1)[0]
            yaw = np.random.uniform(low=yaw_lim[0], high=yaw_lim[1], size=1)[0]
            
            self.robot.set_position([x, y, 0.05])
            quat = T.euler2quat([0.0, 0.0, yaw])
            self.robot.set_orientation(quat)
            print("initialized robot at (pos, yaw, quat)\n", x, y, yaw, quat)

        else:
            pos = [self.initial_robot_pose[0], self.initial_robot_pose[1], 0.05]
            self.robot.set_position(pos)
            quat = T.euler2quat([0.0, 0.0, self.initial_robot_pose[2]])
            self.robot.set_orientation(quat)
            print("initialized robot at (pos, yaw, quat)\n", pos, quat)
        
        # take a few steps
        self.step_sim(10)


    def add_to_objects_dict(self, objects):
        """
        Add objects to self.objects dict
        Args:   
            objects (list): list of objects to add to self.objects
        """
        for obj in objects:
            self.objects[obj.name] = obj

    def check_termination(self):
        """
        Check if termination condition has been met. Overwrite this in child class.

        Returns:
            done (bool): True if termination condition has been met
            info (dict): dictionary of termination conditions
        """
        raise NotImplementedError()
    
    def _check_horizon_reached(self):
        return self.steps >= self.max_steps
    
    def reward(self):
        return 0

    def step(self, action):

        # take environment step
        # print("-------moma wrapper step---------")
        # breakpoint()
        # fix arm joint positions during navigation
        if self.fix_arm_during_nav:
            print("==============WARNING: FIXING ARM DURING NAVIGATION IN MOMA_WRAPPER==============")
            # check if the robot is navigating
            if max(abs(action[self.robot.controller_action_idx["base"]])) > 1e-4:
                control_idx = np.concatenate([self.robot.trunk_control_idx, self.robot.arm_control_idx[self.robot.default_arm]])
                action[self.robot.controller_action_idx["arm_" + self.robot.default_arm]] = self.default_joint_pos[control_idx]
                # pass
        obs, _, _, _ = self.env.step(action) # only use obs from environment - reward, done, info comes from task, but env.task is DummyTask
        
        # get resward
        reward = self.reward() # returns 0 by default

        # get done and info
        done, info = self.check_termination()
        
        self.steps += 1
        return obs, reward, done, info

    def step_sim(self, steps):
        """
        Steps simulation without taking action
        """
        for _ in range(steps):
            og.sim.step()

    def reset(self):
        
        # reset environment
        obs = self.env.reset()

        # # refresh sim
        og.sim.stop()
        og.sim.play()
        # time.sleep(1)
        # reset robot pose
        self.tuck_arm()
        self._initialize_robot_pose()

        # take a few steps 
        self.step_sim(10)

        self.steps = 0

        return obs

    def _check_ontop(self, objA, objB):
        """
        Checks if objA is on top of objB
        """
        # print("-------check_ontop---------")

        return objA.states[OnTop]._get_value(objB)
    
    def _is_gripper_closed(self):
        """
        Checks if gripper is closed
        Args:
            arm (str): "left" or "right" - only works for Tiago right now. not used
        """
        # print("-------is_gripper_closed---------")

        return self.robot.controllers["gripper_" + self.robot.default_arm].is_grasping()
    
    def _is_grasping_obj(self, obj_name):
        """
        Checks if gripper is grasping obj
        """
        # print("-------is_grasping_obj---------")

        obj = self.robot._ag_obj_in_hand[self.robot.default_arm]
        if obj is None:
            return False
        
        return obj.name == obj_name

    def _is_obj_on_floor(self, obj):
        """
        Checks if obj is on floor
        """
        # print("-------is_obj_on_floor---------")
        floor_objects = [self.objects[obj_name] for obj_name in self.objects if "floor" in obj_name]
        for floor_obj in floor_objects:
            if self._check_ontop(obj, floor_obj):
                return True
        return False


    def _check_in_contact(self, pathA, pathB): # just a note. not used
        """
        one way to check if objects are in contact?
        """
        return RigidContactAPI.in_contact([pathA + "/base_link"], [pathB + "/base_link"])
    
    def _teleport_robot(self, pose):
        """
        Teleports robot to specified pose
        Args:
            pose (list): [x, y, yaw]
        """
        quat = T.euler2quat([0.0, 0.0, pose[2]])
        self.robot.set_position([pose[0], pose[1], 0.05])
        self.robot.set_orientation(quat)
        self.step_sim(5)

    def tuck_arm(self):
        if self.raise_hand:
            self.robot.set_joint_positions(self.default_joint_pos)
        else:
            self.robot.tuck()
        # self.step_sim(10)

    def _is_arm_homed(self, arm="all"):
        """
        Checks if arm is in home position.
        Args:
            arm: "left", "right" or "all"
        """
        if self.robot_model == "Tiago":
            if arm == "all":
                control_idx = np.concatenate([self.robot.arm_control_idx["left"], self.robot.arm_control_idx["right"]])
            else:
                control_idx = self.robot.arm_control_idx[arm]        
        
        elif self.robot_model == "Fetch":
            control_idx = self.robot.arm_control_idx[self.robot.default_arm]

        thresh = 0.05
        # print(max(abs(self.robot.get_joint_positions() - self.default_joint_pos)[control_idx]))
        return max(abs(self.robot.get_joint_positions() - self.default_joint_pos)[control_idx]) < thresh

    @property
    def default_joint_pos(self):
        if self.robot_model == "Tiago":
            """
            Tiago joints are defined as the following??? (default tuck position defined in Tiago class - tucked_default_joint_pos)
            [
                0-2: base (3),
                3-4: head (2),
                5-6: torso (1), 0.0?,
                7-9: left arm joint 0, right arm joint 0, 0.0?,
                10-12: left arm joint 1, right arm joint 1, 0.0?,
                13-14: left arm joint 2, right arm joint 2,
                15-16: left arm joint 3, right arm joint 3,
                17-18: left arm joint 4, right arm joint 4,
                19-20: left arm joint 5, right arm joint 5,
                21-22: left arm joint 6, right arm joint 6,
                23-24: left arm joint 7, right arm joint 7,
                25-26: left gripper (2),
                27-28: right gripper (2)
            ]        
            """
            # wheels = [-1.78029833e-04, 3.20231302e-05, -1.85759447e-07]
            # # head = [0.62, -0.76]
            # head = [-1.16488536e-07,   4.55182843e-08]
            # # torso = [0.32, 0.0]
            # torso = [2.36128806e-04,  0.00000000e+00]
            # # left_arm = [-0.61, -1.1, 0.53, 0.98, -1.2, 0.58, 0.0]
            # left_arm = [-0.61, -1.1, 0.87, 1.5, -1.5, 0.45, 0.0]

            # right_arm = [-1.10, 1.47, 2.71, 1.71, -1.57, 1.39, 0.0]
            # arm = [
            #     left_arm[0], right_arm[0], 0.0,
            #     left_arm[1], right_arm[1], 0.0,
            #     left_arm[2], right_arm[2],
            #     left_arm[3], right_arm[3],
            #     left_arm[4], right_arm[4],
            #     left_arm[5], right_arm[5],
            #     left_arm[6], right_arm[6],
            # ]
            # gripper = [0.045] * 4
            # default_q = wheels + head + torso + arm + gripper
            # default_q = np.array(default_q)
            default_q = np.array([
                -1.78029833e-04,  3.20231302e-05, -1.85759447e-07,
                0.0, -0.2,
                0.0,  0.1, -6.10000000e-01,
                -1.10000000e+00,  0.00000000e+00, -1.10000000e+00,  1.47000000e+00,
                0.00000000e+00,  8.70000000e-01,  2.71000000e+00,  1.50000000e+00,
                1.71000000e+00, -1.50000000e+00, -1.57000000e+00,  4.50000000e-01,
                1.39000000e+00,  0.00000000e+00,  0.00000000e+00,  4.50000000e-02,
                4.50000000e-02,  4.50000000e-02,  4.50000000e-02
            ])

        elif self.robot_model == "Fetch":
            """
            default_q = np.array([
                0.0,
                0.0,  # wheels
                0.0,  # trunk
                0.0,
                0.0,
                0.0,  # head
                -0.22184,
                1.53448,
                1.46076,
                -0.84995,
                1.36904,
                1.90996,  # arm
                0.05,
                0.05,  # gripper
            ])
            """
            # this one is free of self collision
            default_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.3, 1.23448, 1.8, -0.15, 1.36904, 1.90996, 0.05, 0.05])

        return default_q
