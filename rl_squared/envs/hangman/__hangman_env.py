from typing import Tuple, Any, Optional, List
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle, seeding

from rl_squared.envs.base_meta_env import BaseMetaEnv

import random
from copy import deepcopy

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
        # max_attempts = kwargs.get('max_attempts', 6)
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

        self._word_list = word_list
        # print(len(self._word_list))
        self._max_length = max_length  # Maximum length for the word representation
        # self._max_attempts = max_attempts  # Maximum number of wrong guesses allowed

        self._observation_space = spaces.Box(
            low=0,                        # Lowest value (0 for masked positions)
            high=27,                      # Highest value (26 for 'z', if 'a' = 1)
            shape=(self._max_length,),    # Only the length of the word
            dtype=np.int32                # Integer representation of each character
        )

        self._action_space = spaces.Discrete(26)  # Guesses 'a' to 'z'

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

        # # print(self._word_list)
        self._word = self.np_random.choice(self._word_list)
        # self._word = 'apple'

        # self.sampled_word.add(self._word)

        # print(self._word)
        
        self._elapsed_steps = 0
        self._episode_reward = 0.0

        # return self._word

    def reset(self, *, seed: Optional[int] = None, \
              options: Optional[dict] = None) -> Tuple:
        """
        Resets the environment and returns the corresponding observation.
        The initial masked word will randomly mask each character position based on logical rules
        considering vowels and rare letters. Previous actions are revealed in the masked word if provided,
        and all instances of a character are consistently masked or unmasked.

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
        # max_word_length = 10  # Define the maximum word length
        word_length = len(self._word)
        self.masked_word = np.zeros(self._max_length, dtype=np.uint8)  # Initialize with zeros for padding

        # Define logical masking rules
        vowels = 'aeiou'
        rare_letters = 'qjxz'
        mask_probabilities = {
            'vowel': 0.8,
            'rare': 0.9,
            'consonant': 0.3
        }

        # Apply a probability to determine if the word should be fully masked or not
        full_mask_probability = 0.1  # Chance to fully mask the word at reset
        apply_full_mask = self._np_random.rand() < full_mask_probability

        character_masking = {}

        for i in range(min(word_length, self._max_length)):
            char = self._word[i].lower()
            if char not in character_masking:  # Decide masking for this character if not already decided
                if char in vowels:
                    mask_probability = mask_probabilities['vowel']
                elif char in rare_letters:
                    mask_probability = mask_probabilities['rare']
                else:
                    mask_probability = mask_probabilities['consonant']

                # Determine if character should be masked or not
                character_masking[char] = self._np_random.rand() < mask_probability or apply_full_mask

            # Apply the determined mask state to the character
            self.masked_word[i] = 27 if character_masking[char] else ord(char) - ord('a') + 1

        # Handle previous actions if specified in options
        if options and 'prev_action' in options:
            prev_actions = options['prev_action']
            for char in prev_actions:
                if char in character_masking:
                    # Unmask all instances of this character
                    indices = [idx for idx, letter \
                               in enumerate(self._word[:self._max_length]) if letter == char]
                    for idx in indices:
                        self.masked_word[idx] = ord(char) - ord('a') + 1

        self.done = False

        # Return the initial observation and an info dictionary
        self._initial_observation = self._get_observation()
        info = {'word': self._word}  # Return the whole word in the info for debugging or logging

        return self._initial_observation, info


    def _get_observation(self):
        """
        Get the current observation of the environment, which now only includes the masked word.
        """
        # Assuming `self.masked_word` is already an array of integers 
        # where each integer represents a masked or visible character.
        # Each character is encoded as an integer, and the masked positions 
        # are denoted by a specific integer (e.g., _ = 27 for masked).
        return deepcopy(self.masked_word)

    def step(self, action):
        assert self.action_space.contains(action), "Action must be between 0 and 25"
        letter = chr(action + ord('a'))
        # print(letter)
        reward = 0

        letter_found = False

        # Check if the guessed letter is in the word and unmask it if so
        for i, char in enumerate(self._word):
            if char == letter and self.masked_word[i] == 27:  # 27 represents a masked position
                self.masked_word[i] = action + 1  # Unmask by storing action index (plus one to avoid zero)
                letter_found = True

        # Assign a positive reward if any correct letter is found, 
        # no penalties for wrong guesses
        if letter_found:
            reward += 1  # Simple reward for finding a letter
        else:
            reward -= 1

        self._elapsed_steps += 1
        self._episode_reward += reward
        
        self.done = True
        terminated = self.done
        truncated = False
        info = {'word': self._word}
        
        if self.done:
            info["episode"] = {}
            info["episode"]["r"] = self._episode_reward
            # if self._auto_reset:
            #     _, _ = self.reset()

        return self._get_observation(), reward, terminated, truncated, info

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        # If you have other stochastic processes, seed them here


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