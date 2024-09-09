from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from src.env import HangmanEnv
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
import gymnasium as gym
from src.model.components import HangmanFeaturesExtractor
from src.utils import read_word_list
import warnings
# from src.data_generation.dataset_analysis import stratified_sample_from_categories, \
#     classify_words_by_unique_letter_count
import numpy as np
import random
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
import multiprocessing
import argparse
from pathlib import Path
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, CallbackList
from tqdm import tqdm  # Import tqdm for progress bars

from src.data_generation.simulation import play_a_game_with_a_word, \
                            simulate_games_for_word_list # testing function
from src.model.inference import guess
from stable_baselines3.common.vec_env import VecNormalize
from src.data_generation.dataset_analysis import *
from stable_baselines3 import DQN
from src.model.callbacks import ChangeWordCallback
from src.model.components.callback import CustomHangmanEvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from typing import Callable

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

warnings.filterwarnings('ignore', category=UserWarning)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def main(args):
    model_dir = Path(args.model_dir)
    log_dir = Path(args.log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)


    # train_list = ['kaw', 'lala', 'ppp', 'howler', 'aludra', 'astays', 'paganists', 'buttering', 'questionnaire', 'shovelman', 'subfunctional', \
    #             'spectrofluorometric', 'neuropsychical', 'lymhpangiophlebitis', 'pseudophilanthropically', 'pseudolamellibranchiate', 'ipil', \
    #             'nunu', 'f', 'wristier', 'resmell', 'lorolla', 'ichthyician', 'sherardized', 'rumenocentesis', 'phallodynia', \
    #             'reflexologically', 'complaintiveness', 'nondiscriminatingly', 'overdogmaticalness', 'predisadvantageously', 'superacknowledgment']

    # train_list = ['mommy', 'obb', 'jj', 'wisped', 'beetiest', 'pokomo', 'quisutsch', 'dramaticism', 'cropseyville', 'nervosity', 'slavoteutonic', \
    #             'adumbratively', 'indemonstrability', 'septocylindrium', 'undiscoverability', 'superacknowledgment']
    # train_list = ['jazz']
    # train_list = ['reflexologically', 'complaintiveness', 'nondiscriminatingly', 'overdogmaticalness', 'predisadvantageously']
    # from src.utils import read_word_list
    # from src.data_generation.dataset_analysis import stratified_sample_from_categories, \
    #     classify_words_by_unique_letter_count, summarize_categories

    # corpus_path = '/media/sayem/510B93E12554BBD1/Hangman/data/words_250000_train.txt'
    # total_samples = 250_000  # This can be adjusted as needed

    # # Read a specified number of words from the corpus
    # corpus = read_word_list(corpus_path, num_samples=total_samples)

    # classification = classify_words_by_unique_letter_count(corpus)
    # # classification = categorize_words_by_length(corpus)
    # summaries = summarize_categories(classification)

    # train_list = classification[16]
    # print(len(train_list))

    train_list = ['kaw']

    # train_list = ['nondiscriminatingly', 'overdogmaticalness', 'predisadvantageously']
    
    # Setup for parallel environments
    num_cpus = multiprocessing.cpu_count()
    print(f"Number of CPU count: {num_cpus}")
    # seeds = [random.randint(0, 10_000) for _ in range(num_cpus)]
    
    N_ENVS = 24

    # Use make_vec_env to create a vectorized environment
    vec_env = make_vec_env(env_id=HangmanEnv, vec_env_cls=SubprocVecEnv, n_envs=N_ENVS)  #
    
    N_STEPS = 200 * len(train_list)

    # n_words_per_batch=1
    # N_STEPS = 20 * n_words_per_batch

    # Setup the PPO model with the training environment
    model = MaskablePPO(MaskableMultiInputActorCriticPolicy, vec_env,  policy_kwargs={
                            "features_extractor_class": HangmanFeaturesExtractor,
                            "features_extractor_kwargs": {"features_dim": 256}}, 
                    verbose=1, device='auto',
                    n_steps=N_STEPS, ent_coef=0.0, learning_rate=linear_schedule(0.0005))

    # # # selected_words = random.sample(val_list, 100)
    # # for train_list in [corpus]:
    eval_callback = CustomHangmanEvalCallback(val_word_list=train_list)
    
    callback = ChangeWordCallback(train_list)

    callbacks = CallbackList([callback, eval_callback])
    
    model.learn(total_timesteps= 20 * (N_STEPS * N_ENVS), progress_bar=True, \
                callback=callbacks)

    # n_words_per_batch=4

    # # Create batches of 10 words until all words are used
    # for i in tqdm(range(0, len(corpus), n_words_per_batch)):
    #     train_list = corpus[i:i + n_words_per_batch]
    #     # print(train_list)

    #     # break

    #     # Initialize callbacks with the current batch of words
    #     eval_callback = CustomHangmanEvalCallback(val_word_list=train_list)
    #     word_change_callback = ChangeWordCallback(train_list)
    #     callbacks = CallbackList([word_change_callback, eval_callback])

    #     # Train the model on this batch of words
    #     model.learn(total_timesteps= 20 * (N_STEPS * N_ENVS), progress_bar=True, callback=callbacks)


    # # model.learn(total_timesteps=args.total_timesteps, progress_bar=True)

    final_model_path = model_dir / "final_hangman_model.zip"
    model.save(str(final_model_path))
    print(f"Model saved to {final_model_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, default='/media/sayem/510B93E12554BBD1/Hangman/data/words_250000_train.txt', help='Path to the corpus file')
    parser.add_argument('--model_dir', type=str, default='/media/sayem/510B93E12554BBD1/Hangman/models/', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='/media/sayem/510B93E12554BBD1/Hangman/logs/', help='Directory to save logs')
    parser.add_argument('--total_samples', type=int, default=250_000, help='Total number of samples to use')
    parser.add_argument('--total_timesteps', type=int, default=1e6, help='Total timesteps for training')
    parser.add_argument('--n_eval_episodes', type=int, default=1_500, help='Number of evaluation episodes')
    # parser.add_argument('--reward_threshold', type=float, default=15.0, help='Reward threshold for early stopping')
    args = parser.parse_args()

    main(args)


# def main(args):
#     model_dir = Path(args.model_dir)
#     log_dir = Path(args.log_dir)
#     model_dir.mkdir(parents=True, exist_ok=True)
#     log_dir.mkdir(parents=True, exist_ok=True)

#     train_list = ['kaw', 'lala', 'ppp', 'howler', 'aludra', 'astays', 'paganists', 'buttering', 'questionnaire', 'shovelman', 'subfunctional', \
#                 'spectrofluorometric', 'neuropsychical', 'lymhpangiophlebitis', 'pseudophilanthropically', 'pseudolamellibranchiate', 'ipil', \
#                 'nunu', 'f', 'wristier', 'resmell', 'lorolla', 'ichthyician', 'sherardized', 'rumenocentesis', 'phallodynia', \
#                 'reflexologically', 'complaintiveness', 'nondiscriminatingly', 'overdogmaticalness', 'predisadvantageously', 'superacknowledgment']

#     val_list = ['mommy', 'obb', 'jj', 'wisped', 'beetiest', 'pokomo', 'quisutsch', 'dramaticism', 'cropseyville', 'nervosity', 'slavoteutonic', \
#                 'adumbratively', 'indemonstrability', 'septocylindrium', 'undiscoverability', 'superacknowledgment']
    
#     # Setup for parallel environments
#     num_cpus = min(multiprocessing.cpu_count(), len(train_list))
#     print(f"Number of CPU count: {num_cpus}")
#     seeds = [random.randint(0, 10_000) for _ in range(num_cpus)]

#     # Distribute the entire training list across multiple CPUs
#     train_env = setup_envs(train_list, num_cpus, seeds, normalize_reward=True)

#     # print(train_env.reset())
    
#     # Setup the PPO model with the training environment
#     model = DQN('MultiInputPolicy', train_env,
#                 policy_kwargs={"features_extractor_class": HangmanFeaturesExtractor,
#                 "features_extractor_kwargs": {"features_dim": 128}},
#                 verbose=0, buffer_size=1_000_000, learning_rate=5e-4)


#     checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=str(model_dir), name_prefix='rl_model')

#     # eval_vec_env = setup_envs(val_list, num_cpus, seeds, normalize_reward=True)


#     eval_callback = EvalCallback(setup_envs(val_list, num_cpus), best_model_save_path=str(model_dir),
#                                          log_path=str(log_dir), eval_freq=5_000,
#                                          deterministic=True, n_eval_episodes=args.n_eval_episodes, verbose=1)

    
#     callbacks = CallbackList([checkpoint_callback, eval_callback])

#     model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)
#     # model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    
#     final_model_path = model_dir / "final_hangman_model.zip"
#     model.save(str(final_model_path))
#     print(f"Model saved to {final_model_path}")