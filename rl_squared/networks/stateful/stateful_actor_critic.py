from typing import Tuple, Optional

import torch
import gymnasium as gym

from rl_squared.networks.base_actor_critic import BaseActorCritic
from rl_squared.networks.stateful.stateful_actor import StatefulActor
from rl_squared.networks.stateful.stateful_critic import StatefulCritic

from rl_squared.networks.base_actor import BaseActor
from rl_squared.networks.base_critic import BaseCritic

class StatefulActorCritic(BaseActorCritic, torch.nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        recurrent_state_size: int,
    ):
        """
        Actor-Critic for a discrete action space.

        Args:
          observation_space (gym.Space): Observation space in 
          which the agent operates.
          action_space (gym.Space): Action space in which the 
          agent operates.
        """
        torch.nn.Module.__init__(self)
        super(StatefulActorCritic, self).__init__(observation_space, \
                                                  action_space)

        self._actor = StatefulActor(
            observation_space=observation_space,
            action_space=action_space,
            recurrent_state_size=recurrent_state_size,
            hidden_sizes=[512],
        )

        self._critic = StatefulCritic(
            observation_space=observation_space,
            recurrent_state_size=recurrent_state_size,
            hidden_sizes=[512],
        )

        self._recurrent_state_size = recurrent_state_size
        self._device = None

    @property
    def actor(self) -> BaseActor:
        """
        Return the actor network.

        Returns:
          BaseActor
        """
        return self._actor

    @property
    def critic(self) -> BaseCritic:
        """
        Return the critic network.

        Returns:
          BaseCritic
        """
        return self._critic

    def to_device(self, device: torch.device) -> "StatefulActorCritic":
        """
        Performs device conversion on the actor and critic.

        Returns:
          StatefulActorCritic
        """
        self._device = device
        self._actor.to(device)
        self._critic.to(device)

        return self
    
    def get_device(self) -> torch.device:
        """
        Get the device on which the neural network components are loaded.

        Returns:
          torch.device: The device on which the model's tensors are stored.
        """
        return self._device or next(self.parameters()).device

    def act(
        self,
        observations: torch.Tensor,  # Expected shape: [batch_size, num_features]
        action_masks: torch.Tensor,  # Optional, shape: [batch_size, num_actions]
        recurrent_states_actor: torch.Tensor,  # Expected shape: [batch_size, hidden_size_actor]
        recurrent_states_critic: torch.Tensor,  # Expected shape: [batch_size, hidden_size_critic]
        recurrent_state_masks: Optional[torch.Tensor] = None,  # Optional, shape: [batch_size, 1] or similar
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a state return the action to take, log probability 
        of said action, and the current state value
        computed by the critic.

        Args:
          observations (torch.Tensor): State in which to take an action.
          recurrent_states_actor (torch.Tensor): Recurrent states for the actor.
          recurrent_states_critic (torch.Tensor): Recurrent states for the critic.
          recurrent_state_masks (torch.Tensor): Masks to be applied to the recurrent states.
          deterministic (bool): Whether to choose actions deterministically.

        Returns:
          Tuple[torch.Tensor, torch.Tensor, \
            torch.Tensor, torch.Tensor, torch.Tensor]
        """
        value_estimate, recurrent_states_critic = self.critic(
            observations, recurrent_states_critic, \
                recurrent_state_masks, self._device
        )

        action_distribution, recurrent_states_actor = self.actor(
            observations, recurrent_states_actor, \
                recurrent_state_masks, self._device
        )

        # print(type(action_distribution))
        
        # print(action_masks)
        # Apply action mask if provided and non-empty
        
        if action_masks is not None and action_masks.any():
            action_distribution.apply_masking(action_masks)

        actions = (
            action_distribution.mode()
            if deterministic
            else action_distribution.sample()
        )

        return (
            value_estimate,
            actions,
            action_distribution.log_prob(actions),
            recurrent_states_actor,
            recurrent_states_critic,
        )
        
    def predict(
        self,
        observations: torch.Tensor,
        action_masks: torch.Tensor,
        recurrent_states_actor: torch.Tensor,
        recurrent_states_critic: torch.Tensor,
        recurrent_state_masks: torch.Tensor = None,
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, \
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform inference on the given state and return the action 
        to take, log probability of said action, and the current state value
        computed by the critic without updating gradients.

        Args:
        observations (torch.Tensor): State in which to take an action.
        recurrent_states_actor (torch.Tensor): Recurrent states for the actor.
        recurrent_states_critic (torch.Tensor): Recurrent states for the critic.
        recurrent_state_masks (torch.Tensor): Masks to be applied to the recurrent states.
        deterministic (bool): Whether to choose actions deterministically (default True for predict).

        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # with torch.no_grad():  # Ensures no gradients are computed during the call
        #     return self.actor(
        #                     observations, recurrent_states_actor,
        #                     recurrent_state_masks, self._device
        #         )

        with torch.no_grad():  # Ensures no gradients are computed during the call
            _, actions, _, recurrent_states_actor, recurrent_states_critic = self.act(
                                observations, action_masks, recurrent_states_actor,
                                recurrent_states_critic, recurrent_state_masks,
                                deterministic
                            )
            
            return actions, recurrent_states_actor, recurrent_states_critic

    def evaluate_actions(
        self,
        inputs: torch.Tensor,
        actions: torch.Tensor,
        recurrent_states_actor: torch.Tensor,
        recurrent_states_critic: torch.Tensor,
        recurrent_state_masks: torch.Tensor = None,
    ) -> Tuple:
        """
        Evaluate actions given observations, states, actions, and recurrent state masks.

        Args:
            inputs (torch.Tensor): Inputs to the actor and the critic.
            actions (torch.Tensor): Actions taken at each timestep.
            recurrent_states_actor (torch.Tensor): Recurrent states for the actor.
            recurrent_states_critic (torch.Tensor): Recurrent states for the critic.
            recurrent_state_masks (torch.Tensor): Masks to be applied to the recurrent states.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        value, _ = self.critic(
            inputs, recurrent_states_critic, recurrent_state_masks, self._device
        )
        dist, _ = self.actor(
            inputs, recurrent_states_actor, recurrent_state_masks, self._device
        )

        log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()

        return value, log_probs, dist_entropy


    def get_value(
        self,
        observations: torch.Tensor,
        recurrent_states_critic: torch.Tensor,
        recurrent_state_masks: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Given a state returns its corresponding value.

        Args:
          observations (torch.Tensor): State in which to take an action.
          recurrent_states_critic (torch.Tensor): Recurrent states that 
          are being used in memory-based policies.
          recurrent_state_masks (torch.Tensor): Masks to be applied to 
          the recurrent states.

        Returns:
          torch.Tensor
        """
        return self.critic(
            observations, recurrent_states_critic, \
                recurrent_state_masks, self._device
        )


    @property
    def recurrent_state_size(self) -> int:
        """
        Returns the size of the encoded state (eg. hidden state in a recurrent agent).

        Returns:
          int
        """
        return self._recurrent_state_size

    def forward(self) -> None:
        """
        Forward pass for the network, in this case not implemented.

        Returns:
          None
        """
        raise NotImplementedError
