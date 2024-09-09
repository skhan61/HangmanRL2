from typing import List, Optional, Tuple, TypeVar, Union
from sb3_contrib.common.maskable.distributions import \
    MaskableDistribution, MaskableCategorical, SelfMaskableCategoricalDistribution, MaybeMasks
import torch
import torch
import random

class MaskableCategoricalDistribution(MaskableDistribution):
    """
    Categorical distribution for discrete actions. Supports invalid action masking.

    :param action_dim: Number of discrete actions
    """
    def __init__(self, action_dim: int):
        super().__init__()
        self.distribution: Optional[MaskableCategorical] = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> torch.nn.Module:
        action_logits = torch.nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(
        self: SelfMaskableCategoricalDistribution, \
        action_logits: torch.Tensor
    ) -> SelfMaskableCategoricalDistribution:
        # Restructure shape to align with logits
        # print(f"action_logits shape: {action_logits.shape}")
        reshaped_logits = action_logits.view(-1, self.action_dim)
        self.distribution = MaskableCategorical(logits=reshaped_logits)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:        
        assert self.distribution is not None, "Must set distribution parameters"
        # return self.distribution.log_prob(actions).log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
        return self.distribution.log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
    
    def sample(self) -> torch.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        sample = self.distribution.sample().unsqueeze(-1)
        return sample

    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.entropy()

    def mode(self) -> torch.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return torch.argmax(self.distribution.probs, dim=-1, keepdim=True)

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def apply_masking(self, masks: MaybeMasks) -> None:
        assert self.distribution is not None, "Must set distribution parameters"
        self.distribution.apply_masking(masks)


def main():
    torch.manual_seed(random.randint(0, 2**32 - 1))
    batch_size = 1
    num_actions = 26

    logits = torch.randn(batch_size, num_actions)
    print(f"Logits: {logits.shape}")
    mask = torch.randint(0, 2, size=(batch_size, num_actions), \
                         dtype=torch.bool)
    print(f"Mask: {mask.shape}")

    # Using MaskableCategoricalDistribution and assuming it has a method proba_distribution
    dist = MaskableCategoricalDistribution(action_dim=num_actions)
    distribution = dist.proba_distribution(action_logits=logits)
    distribution.apply_masking(mask)

    samples = distribution.mode()  # Use mode to get the most likely action per batch
    # print(f"Actions shape:", samples.shape)
    guessed_indices = samples.flatten()

    # Directly verify each guessed index against the mask
    for b, idx in enumerate(guessed_indices):
        if not mask[b, idx]:  # Check if the mask at the batch and action index is False
            guessed_letter = chr(97 + idx.item())
            raise ValueError(f"A disallowed letter was guessed: {guessed_letter}")

    # print("All guessed actions are allowed.")
    # guessed_letters = [chr(97 + idx.item()) for idx in guessed_indices]
    # print("Guessed Letters:", guessed_letters)

if __name__ == "__main__":
    # while True:
    main()