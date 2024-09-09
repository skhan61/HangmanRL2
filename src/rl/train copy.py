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

warnings.filterwarnings('ignore', category=UserWarning)

def env_creator(word_list, seed=None, **kwargs):
    def _init():
        env = HangmanEnv(words=word_list, **kwargs)
        if seed is not None:
            env.seed(seed)
        return Monitor(env)
    return _init

def setup_envs(words, num_envs, seeds=None, normalize_reward=True):
    num_envs = min(num_envs, len(words))
    chunks = np.array_split(np.array(words), num_envs)
    # print(chunks)
    if seeds is None:
        seeds = [None] * num_envs
    envs = [env_creator(chunk.tolist(), seed=seed) for chunk, seed in zip(chunks, seeds)]
    vec_env = SubprocVecEnv(envs)
    if normalize_reward:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_obs=10.0, clip_reward=5.0)
    return vec_env

def main(args):
    model_dir = Path(args.model_dir)
    log_dir = Path(args.log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # # Reading and shuffling the corpus
    # corpus = read_word_list(args.corpus_path, num_samples=args.total_samples)
    # random.shuffle(corpus)

    # # Splitting the corpus into training and validation sets
    # train_size = int(0.8 * args.total_samples)
    # train_list = corpus[:train_size]
    # val_list = corpus[train_size:]
    # print(f"Length of train list: {len(train_list)}")
    # print(f"Length of val list: {len(val_list)}")


    train_list = ['kaw', 'lala', 'ppp', 'howler', 'aludra', 'astays', 'paganists', 'buttering', 'questionnaire', 'shovelman', 'subfunctional', \
                'spectrofluorometric', 'neuropsychical', 'lymhpangiophlebitis', 'pseudophilanthropically', 'pseudolamellibranchiate', 'ipil', \
                'nunu', 'f', 'wristier', 'resmell', 'lorolla', 'ichthyician', 'sherardized', 'rumenocentesis', 'phallodynia', \
                'reflexologically', 'complaintiveness', 'nondiscriminatingly', 'overdogmaticalness', 'predisadvantageously', 'superacknowledgment']

    val_list = ['mommy', 'obb', 'jj', 'wisped', 'beetiest', 'pokomo', 'quisutsch', 'dramaticism', 'cropseyville', 'nervosity', 'slavoteutonic', \
                'adumbratively', 'indemonstrability', 'septocylindrium', 'undiscoverability', 'superacknowledgment']
    
    # Setup for parallel environments
    num_cpus = multiprocessing.cpu_count()
    print(f"Number of CPU count: {num_cpus}")
    seeds = [random.randint(0, 10_000) for _ in range(num_cpus)]

    # Distribute the entire training list across multiple CPUs
    train_env = setup_envs(train_list, num_cpus, seeds, normalize_reward=True)
    
    # Setup the PPO model with the training environment
    model = MaskablePPO(MaskableMultiInputActorCriticPolicy, train_env,
                        policy_kwargs={"features_extractor_class": HangmanFeaturesExtractor,
                                       "features_extractor_kwargs": {"features_dim": 128}},
                        verbose=1, n_steps=2048, ent_coef=0.02)

    # selected_words = random.sample(val_list, 100)
    
    final_results = simulate_games_for_word_list(word_list=val_list, guess_function=guess, \
                                                play_function=play_a_game_with_a_word, \
                                                model=model, solver=None, \
                                                transform=None, process_word_fn=None) 

    # Print overall statistics
    overall_stats = final_results['overall']
    print("\nOverall Statistics:")
    print(f"Total Games: {overall_stats['total_games']}, Wins: {overall_stats['wins']}, Losses: {overall_stats['losses']}")
    print(f"Win Rate: {overall_stats['win_rate']:.2f}, Average_tries_remaining: {overall_stats['average_tries_remaining']:.2f}")
    print()

    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=str(model_dir), name_prefix='rl_model')


    eval_callback = MaskableEvalCallback(setup_envs(val_list, num_cpus), best_model_save_path=str(model_dir),
                                         log_path=str(log_dir), eval_freq=5_000,
                                         deterministic=True, n_eval_episodes=args.n_eval_episodes, verbose=1)

    # stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, verbose=1)
    # eval_callback = MaskableEvalCallback(eval_vec_env, best_model_save_path=str(model_dir),
    #                                     log_path=str(log_dir), eval_freq=5000,
    #                                     deterministic=True, n_eval_episodes=args.n_eval_episodes, 
    #                                     callback_on_new_best=stop_callback,
    #                                     verbose=1)

    
    callbacks = CallbackList([checkpoint_callback, eval_callback])


    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)
    
    # # model.learn(total_timesteps=args.total_timesteps, progress_bar=True)


    final_results = simulate_games_for_word_list(word_list=val_list, guess_function=guess, \
                                                play_function=play_a_game_with_a_word, \
                                                model=model, solver=None, \
                                                transform=None, process_word_fn=None) 

    # Print overall statistics
    overall_stats = final_results['overall']
    print("\nOverall Statistics:")
    print(f"Total Games: {overall_stats['total_games']}, Wins: {overall_stats['wins']}, Losses: {overall_stats['losses']}")
    print(f"Win Rate: {overall_stats['win_rate']:.2f}, Average_tries_remaining: {overall_stats['average_tries_remaining']:.2f}")

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