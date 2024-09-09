from typing import Tuple, Union, List
import gymnasium as gym
from rl_squared.networks.modules.distributions import MaskableCategoricalDistribution

import torch

from rl_squared.networks.modules.distributions import Categorical, DiagonalGaussian
from rl_squared.networks.modules.memory.gru import GRU
from rl_squared.networks.modules.memory.lstm import LSTM
from rl_squared.networks.base_actor import BaseActor
from rl_squared.networks.modules.distributions import (
    FixedGaussian,
    FixedCategorical,
    FixedBernoulli,
)

from rl_squared.utils.torch_utils import init_mlp
from rl_squared.networks.feature_extractor import EmbeddingGRU

class StatefulActor(BaseActor):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        recurrent_state_size: int,
        hidden_sizes: List[int],
    ):
        """
        Stateful actor for a discrete action space.

        Args:
            observation_space (gym.Space): State dimensions for the environment.
            action_space (gym.Space): Action space in which the agent is operating.
            recurrent_state_size (int): Size of the recurrent state.
            hidden_sizes (List[int]): Size of the hidden layers for the policy head.
        """
        super(StatefulActor, self).__init__(observation_space, action_space)

        self._recurrent_state_size = recurrent_state_size

        # modules
        self._feature_extractor = EmbeddingGRU()
        self._gru = GRU(observation_space.shape[0], recurrent_state_size)
        # self._gru = LSTM(observation_space.shape[0], recurrent_state_size)
        self._mlp = init_mlp(recurrent_state_size, hidden_sizes)
        # self._policy_head = self._init_dist(hidden_sizes[-1], action_space)
        self.dist = MaskableCategoricalDistribution(action_dim= action_space.n)
        self._policy_head = self.dist.proba_distribution_net(hidden_sizes[-1])
        # pass

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
    ) -> Tuple[Union[FixedGaussian, FixedBernoulli, FixedCategorical], torch.Tensor]:
        """
        Conduct the forward pass through the network.

        Args:
          x (torch.Tensor): Input for the forward pass.
          recurrent_states (torch.Tensor): Recurrent states for the actor.
          recurrent_state_masks (torch.Tensor): Masks (if any) to be 
          applied to recurrent states.
          device (torch.device): Torch device on which to transfer the tensors.

        Returns:
          Tuple[Categorical, torch.Tensor]
        """
        # print(x)
        # print(x.shape)

        x = self._feature_extractor(x)

        # print(x.shape)
        # print(recurrent_states.shape)
        # print(recurrent_state_masks.shape)

        ## x must be (batch, 116)


        x, recurrent_states = self._gru(
            x, recurrent_states, recurrent_state_masks, \
            device
        )
        x = self._mlp(x) # logits
        # print(x.shape)
        x = self.dist.proba_distribution(self._policy_head(x)) # distribution

        return x, recurrent_states
