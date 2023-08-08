import omnigibson as og
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.usd_utils import RigidContactAPI

from .moma_wrapper import MOMAEnv

import numpy as np
import time

class NavPickEnv(MOMAEnv):
    
    def __init__(
            self,
            name="NavPickEnv",
            max_steps=1000,
            random_init_obj=False, # randomly initialize object position on table
            random_init_base_pose=None, # randomly initialize robot base pose
            random_init_robot_pose=False, # randomly initialize robot base and arm poses - TODO
            initial_robot_pose=[0.0, 0.0, 0.0], # defualt initial pose [x, y, z, yaw] 
            raise_hand=True,
            fix_arm_during_nav=True,
    ):
        config_filename="configs/nav_pick.yaml",

        super().__init__(
            name=name,
            config_filename=config_filename,
            max_steps=max_steps,
            random_init_base_pose=random_init_base_pose,
            random_init_robot_pose=random_init_robot_pose,
            initial_robot_pose=initial_robot_pose,
            raise_hand=raise_hand,
            fix_arm_during_nav=fix_arm_during_nav,
        )

        self.random_init_obj = random_init_obj
        self.grasp_obj_default_pos = [-0.47616568, -1.21954441, 0.5]
        
        self.add_objects()
        self._place_grasp_obj()

        self.step_sim(10)

    def add_objects(self):

        grasp_obj = DatasetObject(
            name="grasp_obj",
            category="cologne",
            model="lyipur",
            scale=0.01
        )
        og.sim.import_object(grasp_obj)

        # update parent class objects dict
        self.add_to_objects_dict([grasp_obj])

    def remove_objects(self):
        grasp_obj = self.objects["grasp_obj"]
        og.sim.remove_object(grasp_obj)

    def _place_grasp_obj(self):

        pos = self._sample_grasp_obj_pos()
        quat = [0,0,1,0]
        obj = self.objects["grasp_obj"]
        obj.set_position_orientation(pos, quat)
        self.step_sim(10)
        obj.set_orientation(quat)
        self.step_sim(10)
        print("Placed object at: ", pos)

    def _sample_grasp_obj_pos(self):
        """
        Sample a pose for grasp_obj
        """
        if not self.random_init_obj:
            # return self.grasp_obj_default_pos
            return np.array([-0.3, -0.8, 0.5])
        
        # delta_pos = np.random.uniform(low=-0.1, high=0.1, size=2)
        delta_pos = np.random.uniform(low=-0.03, high=0.03, size=2)
        pos = np.array([self.grasp_obj_default_pos[0] + delta_pos[0], self.grasp_obj_default_pos[1] + 0.45 + delta_pos[1], self.grasp_obj_default_pos[2]])
        return pos

    def check_termination(self):
        """
        Check if termination condition has been met. Overwrite this in child class.

        Returns:
            done (bool): True if termination condition has been met
            info (dict): dictionary of termination conditions
        """
        # print("-------potato check termination---------")

        success = self.is_success()
        failure, failure_info = self._check_failure()
        done = success or failure
        info = {"success": success, "failure": failure}
        info.update(failure_info)
        return done, info

    def is_success(self):
        """
        Success confition: robot picked up the object
        """
        # print("-------potato check success---------")

        grasping_obj = self._is_grasping_obj("grasp_obj")
        
        return grasping_obj and self._is_arm_homed()

    def _check_failure(self): 
        """
        Failure conditions:
        - object is on the floor
        - max number of steps reached
        """
        # print("-------potato check failure---------")

        horizon_reached = self._check_horizon_reached()
        dropped_obj = self._is_obj_on_floor(self.objects["grasp_obj"])
        failed = dropped_obj or horizon_reached
        failed_info = {
            "dropped_grasp_obj" : dropped_obj,
            "horizon_reached" : horizon_reached,
        }
        return failed, failed_info

    def step(self, action):
        # print("-------potato step---------")

        obs, reward, done, info = super().step(action)

        # TODO - add more to obs?
        return obs, reward, done, info
    
    def reset(self):
        print("RESETTING")
        # reset env and re-add objects
        obs = super().reset()
        self._place_grasp_obj()
        self.step_sim(10)
        return obs
    
    @property
    def env_args(self):
        env_args = {
            "env_name" : self.name,
            "env_type" : 4,
            "env_kwargs" : {
                "max_steps" : self.max_steps,
                "random_init_obj" : self.random_init_obj,
                "random_init_base_pose" : self.random_init_base_pose,
                "initial_robot_pose" : self.initial_robot_pose, 
                "raise_hand" : self.raise_hand,
                "fix_arm_during_nav" : self.fix_arm_during_nav,
            },
        }