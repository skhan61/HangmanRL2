from typing import Tuple, Any, Optional, List
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle, seeding

from rl_squared.envs.base_meta_env import BaseMetaEnv

import random

corpus = ['mommy', 'obb', 'jj', 'wisped', 'beetiest', 'pokomo', 'quisutsch', 'dramaticism', 'cropseyville', \
        'nervosity', 'slavoteutonic', 'adumbratively', 'indemonstrability', 'septocylindrium', 'undiscoverability', \
        'superacknowledgment']


class HangmanEnv(EzPickle, BaseMetaEnv):
    def __init__(
        self, **kwargs
    ):
        """
        Initialize the Hangman environment.

        Args:
            word_list (List[str], optional): List of words. Defaults to ["hello", "world"].
            max_attempts (int, optional): Maximum number of wrong guesses. Defaults to 6.
            max_length (int, optional): Maximum length of the word. Defaults to 35.
            auto_reset (Optional[bool], optional): Automatically reset at end of episode. Defaults to True.
            seed (Optional[int], optional): Random seed. Defaults to None.

        """
        word_list = kwargs.get('word_list', corpus)
        max_attempts = kwargs.get('max_attempts', 6)
        max_length = kwargs.get('max_length', 35)
        auto_reset = kwargs.get('auto_reset', True)
        seed = kwargs.get('seed', 0)

        EzPickle.__init__(self)
        BaseMetaEnv.__init__(self)

        self._auto_reset = auto_reset
        self._elapsed_steps = 0
        self._episode_reward = 0.0
        # self._seed = self.seed(seed)

        # print(self._seed)

        # self._num_actions = num_actions
        self._word = None
        self._vowels = 'aeiou'

        self._word_list = word_list
        # print(len(self._word_list))
        self._max_length = max_length  # Maximum length for the word representation
        self._max_attempts = max_attempts  # Maximum number of wrong guesses allowed

        # self.observation_space = spaces.Dict({
        #     'masked_word': spaces.Box(low=0, high=27, shape=(self._max_length,), \
        #                               dtype=np.uint8),
        #     'guessed_letters': spaces.MultiBinary(26),
        #     'remaining_attempts': spaces.Box(low=0, high=self._max_attempts, \
        #                                      shape=(1,), dtype=np.int64),
        #     'action_mask': spaces.MultiBinary(26),  # Include action masks
        # })

        self.observation_space = spaces.Box(
            low=0,
            high=max(27, self._max_attempts),
            shape=(self._max_length + 26 + 1 + 26,),  # Concatenated arrays
            dtype=np.int32
        )

        self.action_space = spaces.Discrete(26)  # Guesses 'a' to 'z'


    # def _get_observation(self):
    #     # Include the action mask in the observation for compatibility 
    #     # with environments expecting action masks
    #     return {
    #         'masked_word': self.masked_word,
    #         'guessed_letters': self.guessed_letters.astype(np.int32),
    #         'remaining_attempts': self.remaining_attempts,
    #         'action_mask': self.action_mask.astype(np.int32)  # Ensure the data type matches 
    #                                                           # the expectation
    #     }

    def _get_observation(self):
        """
        Get the current observation of the environment in a flattened format.
        """
        # Flatten all parts of the observation into a single array
        return np.concatenate([
            self.masked_word,
            self.guessed_letters.astype(np.int32),
            self.remaining_attempts,
            self.action_mask.astype(np.int32)
        ])

    def sample_task(self) -> None:
        """
        Sample a new bandit task.

        Returns:
            None
        """
        # if self._seed is not None:
        #     self._np_random, seed = seeding.np_random(seed)

        # print(self._seed)

        # available_words = self._word_list - self.sampled_word

        # print(self._word_list)
        self._word = self.np_random.choice(self._word_list)

        # self.sampled_word.add(self._word)

        # print(self._word)
        
        self._elapsed_steps = 0
        self._episode_reward = 0.0

        # return self._word

        # print()
    def action_masks(self) -> np.ndarray:
        """Generate and return current action masks as a numpy array."""
        self.action_mask = 1 - self.guessed_letters.astype(np.uint8)
        return self.action_mask
    
    def reset(
        self, *, seed: Optional[int] = None, \
          options: Optional[dict] = None
    ) -> Tuple:
        """
        Resets the environment and returns the corresponding observation.

        This is different from `sample_task`, unlike the former this will 
        not change the payout probabilities.

        Args:
            seed (int): Random seed.
            options (dict): Additional options.

        Returns:
          Tuple
        """
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        self._elapsed_steps = 0
        self._episode_reward = 0.0

        self.attempts_left = self._max_attempts
        # self.remaining_attempts = self.max_attempts
        self.remaining_attempts = np.array([self._max_attempts], dtype=np.int64)
      
        # self.guessed_letters = np.zeros(26, dtype=np.bool_) #-> must be a set
        self.guessed_letters = np.zeros(26, dtype=np.int64) #-> must be a set
        self.masked_word = np.full(self._max_length, 0, dtype=np.uint8)
        self.done = False

        self.action_mask = np.ones(26, dtype=np.int64)
        # Set initial state based on the word length
        word_length = min(len(self._word), self._max_length)
        self.masked_word[:word_length] = 27  # Assume 27 represents masked letters initially

        # Return the initial observation and an info dictionary
        self._initial_observation = self._get_observation()

        info = {'word': self._word}

        return self._initial_observation, info

    def step(self, action):
        assert self.action_space.contains(action), "Action must be between 0 and 25"
        letter = chr(action + ord('a'))
        reward = 0

        if self.guessed_letters[action]:
            print('Repeating...')
            self._elapsed_steps += 1
            reward -= 50  # Large penalty for repeating a guess
            self._episode_reward += reward
            self.done = True  # Optionally terminate the game due to repetition
            info = {'word': self._word}
            if self.done:
                info["episode"] = {}
                info["episode"]["r"] = self._episode_reward
                # if self._auto_reset:
                #     self.reset()

            return self._get_observation(), reward, self.done, False, info
        
        self._elapsed_steps += 1

        self.guessed_letters[action] = True
        letter_found = False

        # Check if the guessed letter is in the word
        for i, char in enumerate(self._word):
            if char == letter and self.masked_word[i] == 27:
                self.masked_word[i] = ord(letter) - ord('a') + 1
                reward += 1
                letter_found = True

        if not letter_found:
            self.attempts_left -= 1  # Decrement attempts only if the guess is incorrect
            reward -= 1  # Penalty for an incorrect guess
            if self.attempts_left <= 0:
                self.done = True
                reward -= 10  # Large penalty for losing the game

        # Check if the game is over because all letters have been revealed
        if all(self.masked_word[i] != 27 for i in range(len(self._word))):
            self.done = True
            reward += 10  # Large bonus for completing the word

        # Synchronize 'remaining_attempts' in the observation space with 'attempts_left'
        self.remaining_attempts[0] = self.attempts_left

        self._episode_reward += reward

        terminated = self.done
        truncated = False
        info = {'word': self._word}
        self._update_action_mask()
        
        if self.done:
            info["episode"] = {}
            info["episode"]["r"] = self._episode_reward

            # if self._auto_reset:
            #     self.reset()

        # print(reward)

        return self._get_observation(), reward, terminated, truncated, info

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        # If you have other stochastic processes, seed them here

    def _update_action_mask(self):
        """Update and return the action mask where 0 
                     means the action (letter) has been guessed."""
        self.action_mask = 1 - self.guessed_letters.astype(np.uint8)
        return self.action_mask
  
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
        Returns the action space

        Returns:
          Tuple[gym.Space, gym.Space]
        """
        return self.observation_space, self.action_space
    
# import numpy as np
# import copy
# from rl_squared.networks.modules.distributions import MaskableCategoricalDistribution
# # config = {'word_list': corpus, 'max_attempts': 6, \
# #         'max_length': 35, 'auto_reset': True,}

# config = {'word_list': corpus, 'max_length': 35, 'auto_reset': True}

# # Initialize the Hangman environment
# env = HangmanEnv(**config)
# print(env.get_spaces())
# _ = env.seed(20)
# _ = env.sample_task()

# action_dist = MaskableCategoricalDistribution(env.action_space.n)

# # # Reset the environment to start a new game
# # # options = {'prev_action': 'p'}
# options = {}
# initial_observation, info = env.reset(options=options)
# # initial_observation = copy.deepcopy(initial_observation)
# # # print(initial_observation)
# # print(info)
# # # print("Initial observation:", initial_observation)
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
#     observation, reward, done, truncated, info = env.step(action.item())  # Convert tensor to Python int
#     print("Reward:", reward)
#     print("Done:", done)
#     print("Next observation:", observation)
    
#     print("Info:", info)
#     print()

#     initial_observation = copy.deepcopy(observation)

#     if done:
#         # initial_observation, _ = env.reset()
#         break