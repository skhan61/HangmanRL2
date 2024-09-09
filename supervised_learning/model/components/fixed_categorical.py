import torch

class FixedCategorical(torch.distributions.Categorical):
    def sample(self) -> torch.Tensor:
        """
        Sample from the categorical distribution.
        
        Returns:
            torch.Tensor
        """
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Return log probabilities of actions.

        Args:
            actions (torch.Tensor): Actions to be taken in the environment.

        Returns:
            torch.Tensor
        """
        return (
            super().log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self) -> torch.Tensor:
        """
        Returns the mode of the distribution.

        Returns:
            torch.Tensor
        """
        return self.probs.argmax(dim=-1, keepdim=True)

    def apply_mask(self, mask: torch.Tensor) -> None:
        """
        Apply an action mask to the probabilities. This version ensures that the method does not fail
        even if all actions are masked out and avoids dividing by zero effectively.

        Args:
            mask (torch.Tensor): A tensor containing 1s for allowable actions and 0s for disallowed actions,
                                must be the same shape as self.probs.
                                Should be a float tensor to properly handle masking.
        """
        if mask is None:
            raise ValueError("Mask cannot be None")
        
        if mask.shape != self.probs.shape:
            raise ValueError("Mask shape must match the shape of action probabilities.")
        
        # Ensure mask is float to prevent integer multiplication issues
        mask = mask.float()

        # Move the mask to the same device as self.probs to prevent cross-device errors
        mask = mask.to(self.probs.device)

        # Apply the mask and avoid numerical instability
        masked_probs = self.probs * mask
        
        # Sum the probabilities after masking
        masked_probs_sum = masked_probs.sum(dim=-1, keepdim=True)
        
        # Only update probabilities where the sum is greater than zero
        safe_probs = torch.where(
            masked_probs_sum > 0,
            masked_probs / masked_probs_sum,
            self.probs  # Keep the original probabilities if sum is zero to avoid altering them
        )
        
        # Update the probabilities
        self.probs = safe_probs

# import torch

# class FixedCategorical(torch.distributions.Categorical):
#     def sample(self) -> torch.Tensor:
#         """
#         Sample from the categorical distribution.

#         Returns:
#             torch.Tensor
#         """
#         return super().sample().unsqueeze(-1)

#     def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
#         """
#         Return log probs of actions.

#         Args:
#             actions (torch.Tensor): Actions to be taken 
#                     in the environment.

#         Returns:
#             torch.Tensor
#         """
#         return (
#             super()
#             .log_prob(actions.squeeze(-1))
#             .view(actions.size(0), -1)
#             .sum(-1)
#             .unsqueeze(-1)
#         )

#     def mode(self) -> torch.Tensor:
#         """
#         Returns the mode of the distribution.

#         Returns:
#             torch.Tensor
#         """
#         return self.probs.argmax(dim=-1, keepdim=True)


