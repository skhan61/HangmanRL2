import os

import argparse

from rl_squared.training.trainer import ExperimentConfig
from rl_squared.training.trainer import Trainer
from rl_squared.utils.env_utils import register_custom_envs
from src.utils import read_word_list, read_rnd_word_list
from tqdm import tqdm
import random

register_custom_envs()

if __name__ == "__main__":
    corpus_path = 'data/words_250000_train.txt'
    eval_path = 'data/words_250000_train.txt' # just a demo, replace with your hold out set
    # eval_list = read_word_list(corpus_path, num_samples=1_000)
    # corpus = read_word_list(corpus_path, num_samples=250_000)
    train_list = read_rnd_word_list(corpus_path, num_samples=250_000)

    # train_list = list(set(corpus) - set(eval_list))
    chunk_size = 10_000
    num_chunks = len(train_list) // chunk_size
    
    checkpoint_path = None  # Initial checkpoint path is None for the first training session.

    _ = random.shuffle(train_list)
    # checkpoint_path = '/media/sayem/510B93E12554BBD1/Hangman/_results/hangman-v0/run-1725575878/checkpoints/checkpoint-best.pt'

    for i in tqdm(range(num_chunks), desc="Processing Training Chunks"):
        train_subset = train_list[i * chunk_size: (i + 1) * chunk_size]
        # print(len(train_list))
        # Additional code to setup and train using `train_subset`

        eval_list = read_rnd_word_list(eval_path, num_samples=1_000) # play and test games, no effect on training, just for sanity check

        # # print(train_subset)
        # print(f"Loading checkpoint from {checkpoint_path}")

        config = ExperimentConfig(
            algo="PPO",
            env_name="Hangman-v0",
            env_configs={
                'word_list': train_subset,
                'max_attempts': 6,
                'max_length': 35,
                'auto_reset': True
            },
            actor_lr=0.0003,
            critic_lr=0.0001,
            use_linear_lr_decay=True,
            optimizer_eps=0.00001,
            max_grad_norm=0.5,
            random_seed=6785,
            cuda_deterministic=False,
            use_cuda=True,
            policy_iterations=100,
            meta_episode_length=16,
            meta_episodes_per_epoch=len(train_subset),
            num_processes=24,
            discount_gamma=0.99,
            ppo_opt_epochs=15,
            ppo_num_minibatches=25,
            ppo_clip_param=0.1,
            ppo_entropy_coef=0.01,
            ppo_value_loss_coef=0.5,
            use_gae=True,
            gae_lambda=0.3,
            checkpoint_interval=1,
            stagnation_limit = 10
        )

        # trainer.evaluate()

        trainer = Trainer(experiment_config=config,
                    restart_checkpoint=checkpoint_path,
                    eval_list=eval_list)
        
        # print(trainer.config.checkpoint_directory)

        # trainer.evaluate()
        
        # # trainer.evaluate()
        trainer.train(enable_wandb=False)

        # # trainer.evaluate()

        # # Update checkpoint path to the last saved checkpoint for the next chunk
        checkpoint_path = os.path.join(trainer.config.checkpoint_directory, \
                                        "checkpoint-best.pt")
        # checkpoint_path = os.path.join(trainer.config.checkpoint_directory, \
        #                                 "final_stopped.pt")
        # # # print(checkpoint_path)
        print(f"Completed training chunk {i+1}/{num_chunks}")

        # print()

        # # break