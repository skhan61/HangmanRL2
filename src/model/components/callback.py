from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from src.data_generation.simulation import play_a_game_with_a_word, \
                            simulate_games_for_word_list # testing function
from src.model.inference import guess
import os
from stable_baselines3.common.callbacks import BaseCallback


class CustomHangmanEvalCallback(BaseCallback):
    def __init__(self, val_word_list=None):
        
        super().__init__()

        # self.play_function = play_function
        self.val_word_list = val_word_list
        # self.process_word_fn = process_word_fn
        self.best_metric_value = float('-inf')  # Adjust based on what 'best' means


    def _on_step(self):
        """
        Called after every step in the environment.
        """
        # Implement this method to comply with the abstract base class requirement
        return True  # Generally should return True unless you have a condition to stop training

    def _on_rollout_end(self):
        # if self.n_calls % self.eval_freq == 0:
        # Perform the custom evaluation logic
        results = self.evaluate_custom_metrics()
        print("Evaluation results:", results)  # Optional: Log the detailed results for monitoring
        # Decide whether to save the best model based on custom logic
        # if self.should_save_best_model(results):
        #     self.model.save(os.path.join(self.best_model_save_path, 'best_model.zip'))
        return True

    def evaluate_custom_metrics(self):
        """
        Use the simulation function to evaluate the current model.
        """
        # Here you simulate the games using the provided word list and the current model
        results = simulate_games_for_word_list(word_list=self.val_word_list, 
                                               guess_function=guess,
                                               play_function=play_a_game_with_a_word,
                                               model=self.model, 
                                               solver=None, 
                                               transform=None, 
                                               process_word_fn=None)
        
        # You could customize this to return any metric from results you care about
        win_rate = results['overall']['win_rate']
        average_tries_remaining = results['overall']['average_tries_remaining']

        return {'win_rate': win_rate, 'average_tries_remaining': average_tries_remaining}

    def should_save_best_model(self, eval_results):
        """
        Determine if the current model is the best based on win rate.
        """
        current_metric_value = eval_results['win_rate']  # Example criterion
        if current_metric_value > self.best_metric_value:
            self.best_metric_value = current_metric_value
            return True
        return False



# class CustomHangmanEvalCallback(MaskableEvalCallback):
#     def __init__(self, eval_freq=500, 
#                  log_path=None, best_model_save_path=None, 
#                  deterministic=True, 
#                  render=False, verbose=1, 
#                  val_word_list=None):
        
#         super().__init__(eval_env=None, n_eval_episodes=None, eval_freq=eval_freq, 
#                          log_path=log_path, best_model_save_path=best_model_save_path, 
#                          deterministic=deterministic, render=render, verbose=verbose)

#         # self.play_function = play_function
#         self.val_word_list = val_word_list
#         # self.process_word_fn = process_word_fn
#         self.best_metric_value = float('-inf')  # Adjust based on what 'best' means

#     def _on_step(self):
#         if self.n_calls % self.eval_freq == 0:
#             # Perform the custom evaluation logic
#             results = self.evaluate_custom_metrics()
#             print("Evaluation results:", results)  # Optional: Log the detailed results for monitoring
#             # Decide whether to save the best model based on custom logic
#             if self.should_save_best_model(results):
#                 self.model.save(os.path.join(self.best_model_save_path, 'best_model.zip'))
#         return True

#     def evaluate_custom_metrics(self):
#         """
#         Use the simulation function to evaluate the current model.
#         """
#         # Here you simulate the games using the provided word list and the current model
#         results = simulate_games_for_word_list(word_list=self.val_word_list, 
#                                                guess_function=guess,
#                                                play_function=play_a_game_with_a_word,
#                                                model=self.model, 
#                                                solver=None, 
#                                                transform=None, 
#                                                process_word_fn=None)
        
#         # You could customize this to return any metric from results you care about
#         win_rate = results['overall']['win_rate']
#         average_tries_remaining = results['overall']['average_tries_remaining']

#         return {'win_rate': win_rate, 'average_tries_remaining': average_tries_remaining}

#     def should_save_best_model(self, eval_results):
#         """
#         Determine if the current model is the best based on win rate.
#         """
#         current_metric_value = eval_results['win_rate']  # Example criterion
#         if current_metric_value > self.best_metric_value:
#             self.best_metric_value = current_metric_value
#             return True
#         return False


    # def __init__(self, eval_freq=500, 
    #              log_path=None, best_model_save_path=None, 
    #              deterministic=True, 
    #              render=False, verbose=1, 
    #              val_word_list=None):