from typing import Union, Tuple

import numpy as np
import gymnasium as gym
# from gymnasium import spaces
from copy import deepcopy

from gymnasium.envs.registration import EnvSpec

from rl_squared.envs.base_mujoco_meta_env import BaseMujocoMetaEnv
from rl_squared.envs.base_meta_env import BaseMetaEnv


class RLSquaredEnv:
    def __init__(self, env: Union[BaseMetaEnv, BaseMujocoMetaEnv]):
        """
        Abstract class that outlines functions required by 
        an environment for meta-learning via RL-Squared.

        Args:
            env (gym.Env): Environment for general 
            meta-learning around which to add an RL-Squared wrapper.
        """
        self.render_mode = 'human'  # Example default value
        self._wrapped_env = env

        self._action_space = self._wrapped_env.action_space
        self._observation_space = self._make_observation_space()

        # pass these onwards
        self._episode_rewards = 0.0
        self._prev_action = None
        self._prev_reward = None
        self._prev_done = None

        assert isinstance(
            self._observation_space, type(self._wrapped_env.observation_space)
        )

    @property
    def spec(self) -> EnvSpec:
        """
        Returns specs for the environment.

        Returns:
          EnvSpec
        """
        return self._wrapped_env.spec

    def _make_observation_space(self) -> gym.Space:
        """
        Modify the observation space of the 
                wrapped environment to include 
                forward rewards, actions, terminal states.

        Returns:
          gym.Space

        """
        
        obs_dims = gym.spaces.flatdim(self._wrapped_env.observation_space)
        action_dims = gym.spaces.flatdim(self._wrapped_env.action_space)
        new_obs_dims = obs_dims + action_dims + 2
        obs_shape = (new_obs_dims,)

        obs_space = deepcopy(self._wrapped_env.observation_space)
        obs_space._shape = obs_shape

        return obs_space
    
    def action_masks(self) -> np.ndarray:
        """
        Returns the action mask.

        Returns:
          np.ndarray
        """
        return self._wrapped_env.action_masks()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment and return \
            a starting observation.

        Returns:
          np.ndarray
        """
        obs, info = self._wrapped_env.reset()

        if self._prev_action is not None:
            next_obs = self._next_observation(
                obs, self._prev_action, self._prev_reward, self._prev_done
            )
        else:
            next_obs = self._next_observation(obs, None, 0.0, False)

        return next_obs, info

    def step(self, action: Union[int, np.ndarray]) -> Tuple:
        """
        Take a step in the environment.

        Args:
          action (Union[int, np.ndarray]): Action to take in the environment.

        Returns:
          Tuple
        """
        obs, rew, terminated, truncated, info \
            = self._wrapped_env.step(action)
        done = truncated or terminated

        self._prev_action = action
        self._prev_reward = rew
        self._prev_done = done

        next_obs = self._next_observation(
            obs, self._prev_action, self._prev_reward, self._prev_done
        )

        return next_obs, rew, done, info
    
    def _next_observation(
        self, obs: np.ndarray, action: Union[int, np.ndarray], rew: float, done: bool
    ) -> np.ndarray:
        """
        Given an observation, action, reward, and whether an episode is done - return the formatted observation.

        Args:
            obs (np.ndarray): Observation made.
            action (Union[int, np.ndarray]): Action taken in the state.
            rew (float): Reward received.
            done (bool): Whether this is the terminal observation.

        Returns:
            np.ndarray
        """
        # print(obs.shape)
        # print(action)
        # print(rew)
        # print(done)

        if self._wrapped_env.action_space.__class__.__name__ == "Discrete":
            obs = np.concatenate(
                [obs, self._one_hot_action(action), \
                 [rew], [float(done)]]
            )
        else:
            obs = np.concatenate(
                [obs, self._flatten_action(action), \
                 [rew], [float(done)]]
            )

        return obs

    def _flatten_action(self, action: np.ndarray = None) -> np.ndarray:
        """
        In the case of discrete action spaces, this returns a one-hot encoded action.

        Returns:
          np.array
        """
        if action is None:
            flattened = np.zeros(self.action_space.shape[0])
        elif len(action.shape) > 1:
            flattened = action.flatten()
        else:
            flattened = action

        return flattened

    def _one_hot_action(self, action: int = None) -> np.array:
        """
        In the case of discrete action spaces, this returns a one-hot encoded action.

        Returns:
          np.array
        """
        encoded_action = np.zeros(self.action_space.n)

        if action is not None:
            encoded_action[action] = 1.0

        return encoded_action

    @property
    def observation_space(self) -> gym.Space:
        """
        Returns the observation space.

        Returns:
          Union[Tuple, int]
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        """
        Returns the action space.

        Returns:
          int
        """
        return self._wrapped_env.action_space

    def get_spaces(self) -> Tuple[gym.Space, gym.Space]:
        """
        Returns the observation space and action space.

        Returns:
          Tuple[gym.Space, gym.Space]
        """
        return self.observation_space, self.action_space

    def sample_task(self) -> None:
        """
        Samples a new task for the environment.

        Returns:
          np.ndarray
        """
        # reset
        self._prev_action = None
        self._prev_reward = None
        self._prev_done = None

        # sample
        self._wrapped_env.sample_task()
        pass

# from rl_squared.envs.hangman.hangman_env import HangmanEnv
# import numpy as np
# import copy
# from rl_squared.networks.modules.distributions import MaskableCategoricalDistribution
# # config = {'word_list': corpus, 'max_attempts': 6, \
# #         'max_length': 35, 'auto_reset': True,}

# config = {'word_list': corpus, 'max_length': 35, 'auto_reset': True}

# from rl_squared.envs.rl_squared_env import RLSquaredEnv
# env = gym.make('Hangman-v0', **config, disable_env_checker=True)
# env.seed(0)
# env = RLSquaredEnv(env)
# _ = env.sample_task()
# # Reset the environment to start a new game
# initial_observation, _ = env.reset()
# # print("Initial observation:", initial_observation)
# action_dist = MaskableCategoricalDistribution(env.action_space.n)
# print()
# while True:
#     print("Initial observation:", initial_observation)
#     action_mask = env.action_masks()
#     action_mask = torch.from_numpy(action_mask.reshape(1, -1)).type(torch.int32)
#     print("Action mask:", action_mask.shape)

#     dummy_logits = torch.randn(1, 26)  # Use torch.randn instead of np.random.randn
#     print("Dummy logits:", dummy_logits.shape)
#     dist = action_dist.proba_distribution(action_logits=dummy_logits)

#     action_dist.apply_masking(action_mask)
#     action = action_dist.sample()

#     print("Action (index of letter):", action)
#     # Take a step in the environment with the chosen action
#     observation, reward, done, truncated = env.step(action.item())  # Convert tensor to Python int
#     print("Reward:", reward)
#     print("Done:", done)

#     # initial_observation = copy.deepcopy(observation)

#     if done:
#         initial_observation, _ = env.reset()
#         print("Next observation:", initial_observation)
#         break
#     else:
#         print("Next observation:", observation)
#         initial_observation = copy.deepcopy(observation)
    
#     # print("Info:", info)
#     print()
