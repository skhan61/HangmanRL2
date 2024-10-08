from typing import Tuple, List

import numpy as np
import torch.nn as nn
import torch

import gymnasium as gym

from rl_squared.networks.base_critic import BaseCritic

from rl_squared.utils.torch_utils import init_mlp, init_module
from rl_squared.networks.modules.memory.gru import GRU
from rl_squared.networks.modules.memory.lstm import LSTM
from rl_squared.networks.feature_extractor import EmbeddingGRU

class StatefulCritic(BaseCritic):
    def __init__(
        self,
        observation_space: gym.Space,
        recurrent_state_size: int,
        hidden_sizes: List[int],
    ):
        """
        Stateful critic.

        Args:
          observation_space (int): Observation space for the critic.
          recurrent_state_size (int): Size of the recurrent state.
          hidden_sizes (List[int]): Hidden layer sizes for the MLP.
        """
        super(StatefulCritic, self).__init__(observation_space)

        if len(observation_space.shape) != 1:
            raise NotImplementedError("Expected vectorized 1-d observation space.")

        self._hidden_size = hidden_sizes
        self._recurrent_state_size = recurrent_state_size

        self._feature_extractor = EmbeddingGRU()
        self._gru = GRU(observation_space.shape[0], recurrent_state_size)
        # self._gru = LSTM(observation_space.shape[0], recurrent_state_size)
        self._mlp = init_mlp(recurrent_state_size, hidden_sizes)
        self._value_head = init_module(
            nn.Linear(hidden_sizes[-1], 1),
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2),
        )
        pass

    @property
    def recurrent_state_size(self) -> int:
        """
        Return the recurrent state size.

        Returns:
          int
        """
        return self._recurrent_state_size

    def forward(
        self,
        x: torch.Tensor,
        recurrent_states: torch.Tensor,
        recurrent_state_masks: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Conduct the forward pass through the network.

        Args:
          x (torch.Tensor): Input for the forward pass.
          recurrent_states (torch.Tensor): Recurrent states for the actor.
          recurrent_state_masks (torch.Tensor): Masks to be applied to the recurrent state.
          device (torch.device): Device on which to transfer tensors.

        Returns:
          Tuple[torch.Tensor, torch.Tensor]
        """
        x = self._feature_extractor(x)

        # print(x)
        # print(x.shape)

        x, recurrent_states = self._gru(
            x, recurrent_states, recurrent_state_masks, device
        )
        x = self._mlp(x)
        x = self._value_head(x)

        return x, recurrent_states
