# import sys
# sys.path.insert(1, "/home/brainiacx-ws/robomimic")

from copy import deepcopy
from .reaching_env_3d import Franka3DReachingEnv
from .reaching_env_2d import Franka2DReachingEnv

import numpy as np
import gym

import pdb

class RealFrankaEnvSB2D(Franka2DReachingEnv):
    """
    Wrapper for stable-baselines3
    """
    def __init__(
        self,
        maintain_eef_orn=True,
        random_init=False,
        terminate_on_success=True,
        max_episode_steps=200,
        control_freq=2,
    ):

        self.control_scale = 1.0 # scale action by this value

        # define action space and observation state
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1]), # x, y
            high=np.array([1, 1]), # x, y
            shape=(2,),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))

        super().__init__(
            env_name="Franka3DReachingSB",
            control_type="OSC_POSE",
            control_freq=control_freq,
            maintain_eef_orn=maintain_eef_orn,
            random_init=random_init,
            terminate_on_success=terminate_on_success,
            max_episode_steps=max_episode_steps,
        )

    def step(self, action):
        # pdb.set_trace()
        observation, reward, done, info = super().step(action * self.control_scale)
        observation = observation["eef_pos"][:-1]
        return observation, reward, done, info

    def reset(self):
        observation = super().reset()
        return observation["eef_pos"][:-1]

    def _random_init(self, steps=5, xrange=(0, 0.7), yrange=(-1.3, 1.3)):
        return super()._random_init(steps, xrange, yrange)


class RealFrankaEnvSB3D(Franka3DReachingEnv):
    """
    Wrapper for stable-baselines3
    """
    def __init__(
        self,
        maintain_eef_orn=True,
        random_init=False,
        terminate_on_success=True,
        max_episode_steps=200,
        control_freq=2,
    ):

        self.control_scale = 0.70
        # define action space and observation state
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            shape=(3,),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))

        super().__init__(
            env_name="Franka3DReachingSB",
            control_type="OSC_POSE",
            control_freq=control_freq,
            maintain_eef_orn=maintain_eef_orn,
            random_init=random_init,
            terminate_on_success=terminate_on_success,
            max_episode_steps=max_episode_steps,
        )

    def step(self, action):
        observation, reward, done, info = super().step(action * self.control_scale)
        observation = observation["eef_pos"]
        return observation, reward, done, info

    def reset(self):
        observation = super().reset()
        return observation["eef_pos"]

    def _random_init(self, steps=5):
        return super()._random_init(steps)


import robomimic.envs.env_base as EB

class RealFrankaEnv(EB.EnvBase):
    """ 
    Wrapper for robomimic
    """
    def __init__(
        self,
        env_name,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        postprocess_visual_obs=False,
        **kwargs,
    ):

        self._env_name = env_name
        self.env = Franka3DReachingEnv(self._env_name, **kwargs)
        self._init_kwargs = deepcopy(kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.get_observation(observation)
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        return self.get_observation(observation)

    def reset_to(self):
        pass

    def render(self):
        return None

    def get_observation(self, obs):
        return obs

    def get_state(self):
        return None

    def get_reward(self):
        self.env._get_reward()

    def get_goal(self):
        """Return center of target"""
        return [np.mean(np.array(self.env.target_x)), np.mean(np.array(self.env.target_y)), np.mean(np.array(self.env.target_z))]

    def set_goal(self):
        pass

    def is_done(self):
        print("done? ", self.env_is_done)
        return self.env._is_done()

    def is_success(self):

        success = self.env._is_success()
        if isinstance(success, dict):
            assert "task" in success
            return success

        return {"task" : success}

    @property
    def action_dimension(self):
        return self.env.action_dimension
        
    @property
    def name(self):
        return self._env_name

    @property
    def type(self):
        return EB.EnvType.CUSTOM_REALFRANKA_TYPE

    def serialize(self):
        return dict(env_name=self._env_name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))

    @property
    def rollout_exceptions(self):
        return None
   
    @classmethod
    def create_for_data_processing(
        cls,
        env_name,
        camera_names,
        camera_height,
        camera_width,
        reward_shaping,
        **kwargs
    ):
        return cls(
            env_name=env_name,
            render=False, 
            render_offscreen=False, 
            use_image_obs=False, 
            postprocess_visual_obs=False,
            **kwargs,
        )
