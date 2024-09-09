from itertools import cycle
from stable_baselines3.common.callbacks import BaseCallback

class ChangeWordCallback(BaseCallback):
    """
    This callback changes the word in the Hangman environment at each episode reset.
    It cycles through a list of words, setting a new word for each new episode.
    """
    def __init__(self, word_list):
        super().__init__()
        # Create an iterator from the word list
        self.words = cycle(word_list)
        self.counter = 0

    def _on_training_start(self):
        """
        Called before the first gradient step.
        """
        # Set the initial word when training starts
        self.training_env.env_method("set_word", next(self.words))

    def _on_rollout_start(self):
        """
        Called at the beginning of each rollout, reset the environment's word here.
        """
        # if self.counter > 5:
        #     self.counter = 0
        # Set a new word from the list at the start of each new episode
        self.training_env.env_method("set_word", next(self.words))

    def _on_rollout_end(self) -> None:
        self.counter += 1

    def _on_step(self):
        """
        Called after every step in the environment.
        """
        # Implement this method to comply with the abstract base class requirement
        return True  # Generally should return True unless you have a condition to stop training

