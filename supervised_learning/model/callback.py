from lightning.pytorch.callbacks import Callback

from supervised_learning.model.simulation import play_a_game_with_a_word, \
                            simulate_games_for_word_list, guess # testing function
import torch

class SimulationCallback(Callback):
    def __init__(self, word_list):
        self.word_list = word_list
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Ensure the model is in evaluation mode and no grad is applied
        pl_module.eval()
        # # print('here')
        with torch.no_grad():
            final_results = simulate_games_for_word_list(
                word_list=self.word_list, 
                guess_function=guess, 
                play_function=play_a_game_with_a_word, 
                model=pl_module
            )

            # Print overall statistics
            overall_stats = final_results['overall']
            print("Overall Statistics from Validation Epoch End:")
            print(f"Total Games: {overall_stats['total_games']}, Wins: {overall_stats['wins']}, Losses: {overall_stats['losses']}")
            print(f"Win Rate: {overall_stats['win_rate']:.2f}, Average Tries Remaining: {overall_stats['average_tries_remaining']:.2f}")

            # Optionally you can also log these results using the logger
            if trainer.logger:
                trainer.logger.log_metrics({
                    "win_rate": overall_stats['win_rate'],
                    "average_tries_remaining": overall_stats['average_tries_remaining']
                })

            # print("Sanity Check: Validation Epoch End Completed")


# # Usage
# trainer = Trainer(
#     callbacks=[
#         SimulationCallback(
#             word_list=word_list, 
#             guess_function=guess, 
#             play_function=play_a_game_with_a_word, 
#             model=model
#         )
#     ]
# )

