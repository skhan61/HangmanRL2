import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from src.model.components import HangmanFeaturesExtractor

class HangmanSACPolicyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # Print debug information
        # print(f"Initializing CustomTorchModel with num_outputs: {num_outputs}")
        # print(f"obs_space: {obs_space}, action_space: {action_space}")
        TorchModelV2.__init__(self, obs_space, action_space, \
                              None, model_config, name)
        nn.Module.__init__(self)

        # print(obs_space)
        self.feature_extractor \
            = HangmanFeaturesExtractor(features_dim=128)
        self.policy_net = nn.Linear(128, num_outputs)
        # Initialize the current_observation buffer
        self.current_observation = None  # Initialize the current observation holder
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):

        # print("input dict from policy model", input_dict)
        # print(input_dict['obs']['masked_word'].shape)
        # print(input_dict['obs']['action_mask'].shape)
        # print(input_dict['obs']['guessed_letters'].shape)

        features = self.feature_extractor(input_dict['obs'])
        action_mask = input_dict['obs']['action_mask']
        logits = self.policy_net(features)

        # Apply action mask using a safer large negative value
        inf_mask = (action_mask == 0).float() * -1e9
        masked_logits = logits + inf_mask

        return masked_logits, state
    
    # @override(TorchModelV2)
    # def forward(self, input_dict, state, seq_lens):
    #     # print("input dict from policy model", input_dict)
    #     features = self.feature_extractor(input_dict['obs'])
    #     action_mask = input_dict['obs']['action_mask']
    #     # print(action_mask)
    #     logits = self.policy_net(features)

    #     # Correctly apply action mask through multiplication
    #     # masked_logits = logits * (action_mask.float())  # Ensure action_mask is a float for multiplication
    #     # Apply masking
    #     # Apply the correct transformation to the mask
    #     transformed_mask = (1 - action_mask) * -1e9

    #     # Apply the transformed mask to the logits
    #     masked_logits = logits + transformed_mask
    #     return masked_logits, state

        # return logits, state
    
    # def get_current_observation(self):
    #     return self.current_observation

class HangmanSACQModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, None, model_config, name)
        nn.Module.__init__(self)
        self.feature_extractor \
            = HangmanFeaturesExtractor(features_dim=128)
        self.policy_net = nn.Linear(128, num_outputs)
        # print("Model initialized with:")
        # print(f"Observation Space: {obs_space}")
        # print(f"Action Space: {action_space}")
        # print(f"Number of Outputs: {num_outputs}")
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        features = self.feature_extractor(input_dict['obs'])
        action_mask = input_dict['obs']['action_mask']
        logits = self.policy_net(features)

        # Apply action mask using a safer large negative value
        inf_mask = (action_mask == 0).float() * -1e9
        masked_logits = logits + inf_mask

        return masked_logits, state
    

    # @override(TorchModelV2)
    # def forward(self, input_dict, state, seq_lens):
    #     features = self.feature_extractor(input_dict['obs'])
    #     action_mask = input_dict['obs']['action_mask']
    #     # print(action_mask)
    #     logits = self.policy_net(features)

    #     # # Correctly apply action mask through multiplication
    #     # masked_logits = logits * (action_mask.float())  # Ensure action_mask is a float for multiplication
    #     # Apply masking
    #     # Apply the correct transformation to the mask
    #     transformed_mask = (1 - action_mask) * -1e9

    #     # Apply the transformed mask to the logits
    #     masked_logits = logits + transformed_mask
        
    #     return masked_logits, state



# class HangmanSACQModel(TorchModelV2, nn.Module):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         # Print debug information
#         # print(f"Initializing CustomTorchModel with num_outputs: {num_outputs}")
#         # print(f"obs_space: {obs_space}, action_space: {action_space}")
#         TorchModelV2.__init__(self, obs_space, action_space, None, model_config, name)
#         nn.Module.__init__(self)
#         self.feature_extractor = HangmanFeaturesExtractor(features_dim=128)
#         self.policy_net = nn.Linear(128, num_outputs)
#         # Initialize the current_observation buffer
#         # self.current_observation = None  # Initialize the current observation holder

#     @override(TorchModelV2)
#     def forward(self, input_dict, state, seq_lens):
#         features = self.feature_extractor(input_dict['obs'])
#         action_mask = input_dict['obs']['action_mask']
#         logits = self.policy_net(features)

#         # Apply action mask using a safer large negative value
#         inf_mask = (action_mask == 0).float() * -1e9
#         masked_q_values = logits + inf_mask

#         return masked_q_values, state
    


# import numpy as np
# import torch

# # Number of samples in the batch
# batch_size = 5

# # Initialize the batch dictionary with custom data, converting to PyTorch tensors
# input_dict = {
#     "obs": {
#         "action_mask": torch.from_numpy(np.ones((batch_size, 26), dtype=np.int32)).float(),
#         "guessed_letters": torch.from_numpy(np.zeros((batch_size, 26), dtype=np.int32)).float(),
#         "masked_word": torch.from_numpy(np.zeros((batch_size, 35), dtype=np.uint8)).long(),
#         "remaining_attempts": torch.from_numpy(np.full((batch_size, 1), 6, dtype=np.int32)).float()
#     }
# }

# # Customizing data for one specific case to mimic the initial example
# # Set all 'action_mask' for letter 'j' to 0 to simulate that it has been guessed
# input_dict["obs"]["action_mask"][:, 9] = 0
# # Set 'guessed_letters' for letter 'j' to 1 (True) to indicate it's been guessed
# input_dict["obs"]["guessed_letters"][:, 9] = 1
# # Modify 'masked_word' to reveal the letter 'j' at a specific position in the word
# input_dict["obs"]["masked_word"][:, 9] = ord('j') - ord('a')  # assuming 'a' is 0, 'b' is 1, etc.

# input_dict


# import ray
# from ray.rllib.algorithms.dqn import DQNConfig
# from ray.rllib.algorithms.sac import SACConfig
# from src.model import HangmanSACPolicyModel, HangmanSACQModel
# torch.autograd.set_detect_anomaly(True)


# policy_model = HangmanSACPolicyModel(env.observation_space, env.action_space, \
#                               num_outputs=26, model_config={}, name=None)

# q_model = HangmanSACQModel(env.observation_space, env.action_space, \
#                               num_outputs=26, model_config={}, name=None)