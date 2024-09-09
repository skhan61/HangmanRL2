import os

import torch
import wandb

import rl_squared.utils.logging_utils as logging_utils
from rl_squared.training.experiment_config import ExperimentConfig
from rl_squared.learners.ppo import PPO

from rl_squared.utils.env_utils import make_vec_envs
from rl_squared.utils.training_utils import (
                                    sample_meta_episodes,
                                    save_checkpoint,
                                    timestamp,
                                )
from tqdm import tqdm

from rl_squared.training.meta_batch_sampler import MetaBatchSampler
from rl_squared.networks.stateful.stateful_actor_critic import StatefulActorCritic
from src.data_generation.simulation import play_a_game_with_a_word, simulate_games_for_word_list
from src.model.inference import guess #, guess_character
from src.data_generation.simulation import play_a_game_with_a_word, \
                            simulate_games_for_word_list # testing function
from src.utils import read_word_list
from typing import List

import gc

def create_stopping_criterion(patience, checkpoint_dir, \
                            checkpoint_interval, optimizer, \
                            actor_critic, config):
    best_value = float('-inf')
    stagnation_count = 0

    def stopping_criterion(metric_value, current_iteration):
        nonlocal best_value, stagnation_count
        is_last_iteration = current_iteration == (config.policy_iterations - 1)
        checkpoint_name = "best" if current_iteration % checkpoint_interval == 0 or is_last_iteration else None

        if metric_value > best_value:
            best_value = metric_value
            stagnation_count = 0  # Reset stagnation count

            # Save the best model checkpoint
            if checkpoint_name:
                save_checkpoint(
                    iteration=current_iteration,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_name=checkpoint_name,
                    actor_critic=actor_critic,
                    optimizer=optimizer,
                )
                print(f"New best model saved with performance: {metric_value} at iteration {current_iteration}")
            return 0  # Continue training

        else:
            stagnation_count += 1
            if stagnation_count >= patience:
                print("Stopping training due to no improvement.")
                return 1  # Stop training

            if checkpoint_name:
                save_checkpoint(
                    iteration=current_iteration,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_name="last",  # Regular checkpoint save
                    actor_critic=actor_critic,
                    optimizer=optimizer,
                )
            return 0  # Continue training

    return stopping_criterion


class Trainer:
    def __init__(
        self, experiment_config: ExperimentConfig, restart_checkpoint: str = None,
        eval_list: List[str] = None
    ):
        """
        Initialize an instance of a trainer for PPO.

        Args:
            experiment_config (ExperimentConfig): Params to be used for the trainer.
            restart_checkpoint (str): Checkpoint path from where to restart the experiment.
        """
        self.config = experiment_config

        # private
        self._device = None
        self._log_dir = None

        # restart
        self._restart_checkpoint = restart_checkpoint

        # corpus_path = '/media/sayem/510B93E12554BBD1/Hangman/data/20k.txt'
        self._word_list = eval_list

        self.rl_squared_envs = make_vec_envs(
            self.config.env_name,
            self.config.env_configs,
            self.config.random_seed,
            self.config.num_processes,
            self.device,
        )

        self.actor_critic = StatefulActorCritic(
            self.rl_squared_envs.observation_space,
            self.rl_squared_envs.action_space,
            recurrent_state_size=1024,
        ).to_device(self.device)

        self.best_win_rate = 0
        self.no_improvement_count = 0
        # self.stagnation_limit = 10  # Evaluations without improvement before stopping

    def simulate_games(self):
        """
        Evaluate the model and decide whether to continue training based on the win rate.
        Returns 1 if training should stop due to no improvement in win rate, otherwise 0.
        """
        self.actor_critic.eval()  # Set the model to evaluation mode

        results = simulate_games_for_word_list(word_list=self._word_list, guess_function=guess, 
                                               play_function=play_a_game_with_a_word, 
                                               model=self.actor_critic)
        stats = results['overall']
        print("Overall Statistics:")
        print(f"Total Games: {stats['total_games']}, Wins: {stats['wins']}, Losses: {stats['losses']}")
        print(f"Win Rate: {stats['win_rate']:.2f}, Average Tries Remaining: {stats['average_tries_remaining']:.2f}")

        current_win_rate = stats['win_rate']
        if current_win_rate > self.best_win_rate:
            self.best_win_rate = current_win_rate
            self.no_improvement_count = 0  # Reset count on improvement
        else:
            self.no_improvement_count += 1  # Increment count on no improvement

        self.actor_critic.train()  # Reset model to training mode

        if self.no_improvement_count >= self.config.stagnation_limit:
            # print(f"Stopping training due to no improvement in win rate for {self.config.stagnation_limit} evaluations.")
            return 1  # Signal to stop training
        
        return 0  # Signal to continue training


    def train(
        self,
        is_dev: bool = True,
        enable_wandb: bool = True,
    ) -> None:
        """
        Train an agent based on the configs specified by the training parameters.

        Args:
            is_dev (bool): Whether to log the run statistics as a `dev` run.
            enable_wandb (bool): Whether to log to Wandb, `True` by default.

        Returns:
            None
        """
        # log
        self.save_params()

        if enable_wandb:
            wandb.login()
            project_suffix = "-dev" if is_dev else ""
            wandb.init(project=f"rl-squared{project_suffix}", \
                       config=self.config.dict)

        # seed
        torch.manual_seed(self.config.random_seed)
        torch.cuda.manual_seed_all(self.config.random_seed)

        # clean
        logging_utils.cleanup_log_dir(self.log_dir)

        torch.set_num_threads(1)

        _ = self.actor_critic.train()

        ppo = PPO(
            actor_critic=self.actor_critic,
            clip_param=self.config.ppo_clip_param,
            opt_epochs=self.config.ppo_opt_epochs,
            num_minibatches=self.config.ppo_num_minibatches,
            value_loss_coef=self.config.ppo_value_loss_coef,
            entropy_coef=self.config.ppo_entropy_coef,
            actor_lr=self.config.actor_lr,
            critic_lr=self.config.critic_lr,
            eps=self.config.optimizer_eps,
            max_grad_norm=self.config.max_grad_norm,
        )

        current_iteration = 0

        # load
        if self._restart_checkpoint:
            print(f"Loading checkpoint from {self._restart_checkpoint}")
            checkpoint = torch.load(self._restart_checkpoint, \
                                    map_location=self.device, \
                                    weights_only=True)
            # current_iteration = checkpoint["iteration"]
            # actor_critic.actor.load_state_dict(checkpoint["actor"])
            # actor_critic.critic.load_state_dict(checkpoint["critic"])
            self.actor_critic.load_state_dict(checkpoint["actor_critic"])
            ppo.optimizer.load_state_dict(checkpoint["optimizer"])
        
        else:
            print("No checkpoint found or provided, starting training from scratch.")

        stopping_criterion = create_stopping_criterion(
                        patience=10,
                        checkpoint_dir=self.config.checkpoint_directory,
                        checkpoint_interval=self.config.checkpoint_interval,
                        optimizer=ppo.optimizer,
                        actor_critic=self.actor_critic,
                        config=self.config
                    )
        
        for j in tqdm(range(current_iteration, self.config.policy_iterations),
                      desc="Policy Iterations", leave=False):

            torch.cuda.empty_cache()  # Clear CUDA cache
            gc.collect()  # Trigger garbage collection to release unreferenced memory

            # Annealing learning rates if applicable
            if self.config.use_linear_lr_decay:
                ppo.anneal_learning_rates(j, self.config.policy_iterations)

            # Sampling meta episodes
            meta_episode_batches, meta_train_reward_per_step = sample_meta_episodes(
                self.actor_critic,
                self.rl_squared_envs,
                self.config.meta_episode_length,
                self.config.meta_episodes_per_epoch,
                self.config.use_gae,
                self.config.gae_lambda,
                self.config.discount_gamma,
                self.device,
            )

            minibatch_sampler = MetaBatchSampler(meta_episode_batches, self.device)
            ppo_update = ppo.update(minibatch_sampler)

            # Log updates
            wandb_logs = {
                "meta_train/mean_policy_loss": ppo_update.policy_loss,
                "meta_train/mean_value_loss": ppo_update.value_loss,
                "meta_train/mean_entropy": ppo_update.entropy,
                "meta_train/approx_kl": ppo_update.approx_kl,
                "meta_train/clip_fraction": ppo_update.clip_fraction,
                "meta_train/explained_variance": ppo_update.explained_variance,
                "meta_train/mean_meta_episode_reward": meta_train_reward_per_step \
                    * self.config.meta_episode_length,
            }

            # Print training updates
            print("Training Update:")
            for key, value in wandb_logs.items():
                print(f"{key.split('/')[-1]}: {value:.4f}")

            # # Checkpointing
            # is_last_iteration = j == (self.config.policy_iterations - 1)
            # checkpoint_name = str(timestamp()) if self.config.checkpoint_all else "last"
            # if j % self.config.checkpoint_interval == 0 or is_last_iteration:
            #     save_checkpoint(
            #         iteration=j,
            #         checkpoint_dir=self.config.checkpoint_directory,
            #         checkpoint_name=checkpoint_name,
            #         actor_critic=self.actor_critic,
            #         optimizer=ppo.optimizer,
            #     )

            _ = self.simulate_games()

            # metric_value = ppo_update.explained_variance # ppo_update.policy_loss  # Example metric
            metric_value = wandb_logs["meta_train/mean_meta_episode_reward"]
            if stopping_criterion(metric_value, j):
                break  # Stop training if criterion returns 1

            # Log to WandB
            if enable_wandb:
                wandb.log(wandb_logs)

            # _ = self.simulate_games()
            print()

            # After each training update
            torch.cuda.empty_cache()  # Optionally clear cache again after an update
            gc.collect()

        # Clean up WandB session if training completes normally
        if enable_wandb and j != (self.config.policy_iterations - 1):
            wandb.finish()

    @property
    def log_dir(self) -> str:
        """
        Returns the path for training logs.

        Returns:
            str
        """
        if not self._log_dir:
            self._log_dir = os.path.expanduser(self.config.log_dir)

        return self._log_dir

    def save_params(self) -> None:
        """
        Save experiment_config to the logging directory.

        Returns:
          None
        """
        self.config.save()
        pass

    @property
    def device(self) -> torch.device:
        """
        Torch device to use for training and optimization.

        Returns:
          torch.device
        """
        if isinstance(self._device, torch.device):
            return self._device

        use_cuda = self.config.use_cuda and torch.cuda.is_available()
        if use_cuda and self.config.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self._device = torch.device("cuda:0" if use_cuda else "cpu")

        return self._device


    
    # def evaluate(self):
    #     """
    #     Evaluate the model by playing games with the trained model and logging the outcomes.

    #     Args:
    #         model: The model to evaluate.

    #     Returns:
    #         None
    #     """

    #     self.actor_critic.eval()  # Set the model to evaluation mode

    #     final_results = simulate_games_for_word_list(word_list=self._word_list, guess_function=guess, \
    #                                                  play_function=play_a_game_with_a_word, \
    #                                                  model=self.actor_critic)

    #     overall_stats = final_results['overall']
    #     print("Overall Statistics:")
    #     print(f"Total Games: {overall_stats['total_games']}, Wins: {overall_stats['wins']}, Losses: {overall_stats['losses']}")
    #     print(f"Win Rate: {overall_stats['win_rate']:.2f}, Average Tries Remaining: {overall_stats['average_tries_remaining']:.2f}")

    #     self.actor_critic.train()  # Reset model to training mode
