from typing import Tuple, Any, Optional
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle, seeding

from rl_squared.envs.base_meta_env import BaseMetaEnv


class NavigationEnv(EzPickle, BaseMetaEnv):
    def __init__(
        self,
        episode_length: int,
        low: float = -0.5,
        high: float = 0.5,
        auto_reset: bool = True,
        seed: Optional[int] = None,
    ):
        """
        2D navigation problems, as described in [1].

        The code is adapted from https://github.com/cbfinn/maml_rl/

        At each time step, the 2D agent takes an action (its velocity, clipped in [-0.1, 0.1]), and receives a penalty
        equal to its L2 distance to the goal position (ie. the reward is `-distance`).

        The 2D navigation tasks are generated by sampling goal positions from the uniform distribution on [-0.5, 0.5]^2.

        [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic Meta-Learning for Fast Adaptation of Deep
        Networks", 2017 (https://arxiv.org/abs/1703.03400)

        Args:
            episode_length (int): Episode length for the navigation environment.
            low (float): Lower bound for the x & y positions.
            high (float): Upper bound for the x & y positions.
            auto_reset (bool): Whether to auto-reset after step limit.
            seed (int): Random seed.
        """
        EzPickle.__init__(self)
        BaseMetaEnv.__init__(self, seed)

        self.viewer = None
        self._episode_length = episode_length
        self._auto_reset = auto_reset

        self._elapsed_steps = 0
        self._episode_reward = 0.0

        self._num_dimensions = 2
        self._start_state = np.zeros(self._num_dimensions, dtype=np.float32)

        self._low = low
        self._high = high

        # sampled later
        self._current_state = None
        self._goal_position = None

        # spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

        # sample
        self.sample_task()
        pass

    def sample_task(self) -> None:
        """
        Sample a new goal position for the navigation task

        Returns:
            None
        """
        self._current_state = self._start_state
        self._elapsed_steps = 0
        self._episode_reward = 0.0
        self._goal_position = self.np_random.uniform(self._low, self._high, size=2)
        pass

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple:
        """
        Resets the environment and returns the corresponding observation.

        Args:
            seed (int): Random seed.
            options (dict): Additional options.

        Returns:
            Tuple
        """
        if seed is not None:
            self.np_random, seed = seeding.np_random(seed)

        self._current_state = self._start_state
        self._elapsed_steps = 0
        self._episode_reward = 0.0

        return self._current_state, {}

    def step(self, action: np.ndarray) -> Tuple:
        """
        Take a step in the environment and return the corresponding observation, action, reward, plus additional info.

        Args:
            action (np.ndarray): Action to be taken in the environment.

        Returns:
            Tuple
        """
        self._elapsed_steps += 1
        action = np.clip(action, -0.1, 0.1)

        assert self.action_space.contains(action)
        self._current_state = self._current_state + action

        x_dist = self._current_state[0] - self._goal_position[0]
        y_dist = self._current_state[1] - self._goal_position[1]

        reward = -np.sqrt(x_dist**2 + y_dist**2)
        self._episode_reward += reward

        terminated = (np.abs(x_dist) < 0.01) and (np.abs(y_dist) < 0.01)
        truncated = self.elapsed_steps == self.max_episode_steps
        done = truncated or terminated

        info = {}
        if done:
            info["episode"] = {}
            info["episode"]["r"] = self._episode_reward

            if self._auto_reset:
                observation, _ = self.reset()
                pass

        return self._current_state, reward, terminated, truncated, info

    @property
    def observation_space(self) -> gym.Space:
        """
        Returns the observation space of the environment.

        Returns:
            gym.Space
        """
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value: Any) -> None:
        """
        Set the observation space for the environment.

        Returns:
            gym.Space
        """
        self._observation_space = value

    @property
    def action_space(self) -> gym.Space:
        """
        Returns the action space

        Returns:
            gym.Space
        """
        return self._action_space

    @action_space.setter
    def action_space(self, value: Any) -> None:
        """
        Set the action space for the environment.

        Returns:
            gym.Space
        """
        self._action_space = value

    def get_spaces(self) -> Tuple[gym.Space, gym.Space]:
        """
        Get the observation space and the action space.

        Returns:
            Tuple
        """
        return self._observation_space, self._action_space

    @property
    def elapsed_steps(self) -> int:
        """
        Returns the elapsed number of episode steps in the environment.

        Returns:
            int
        """
        return self._elapsed_steps

    @property
    def max_episode_steps(self) -> int:
        """
        Returns the maximum number of episode steps in the environment.

        Returns:
          int
        """
        return self._episode_length

    def render(self, mode: str = "human") -> None:
        """
        Render the environment given the render mode.

        Args:
            mode (str): Mode in which to render the environment.

        Returns:
            None
        """
        pass

    def close(self) -> None:
        """
        Close the environment.

        Returns:
            None
        """
        pass
