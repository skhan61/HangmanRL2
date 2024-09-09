import numpy as np
from typing import Tuple, List

import torch

from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv

def to_tensor(item, device):
    """
    Recursively converts numpy arrays to torch tensors and sends them to the specified device.

    Args:
        item: The item to convert (could be a numpy array, list, or dictionary).
        device: The torch device to which the tensors should be sent.

    Returns:
        The converted item, with all numpy arrays converted to torch tensors on the specified device.
    """
    if isinstance(item, np.ndarray):
        return torch.from_numpy(item).float().to(device)
    elif isinstance(item, dict):
        return {k: to_tensor(v, device) for k, v in item.items()}
    elif isinstance(item, list):
        return [to_tensor(x, device) for x in item]
    return item  # Return the item as is if it's not a list, dict, or ndarray


import numpy as np
from typing import Tuple, List

import torch

from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv


class PyTorchVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, device: torch.device):
        """
        Initialize an environment compatible with PyTorch.

        Args:
            venv (VecEnv): Vectorized environment to provide 
                a PyTorch wrapper for.
            device (torch.device): Device for PyTorch tensors.
        """
        super(PyTorchVecEnvWrapper, self).__init__(venv)
        self.device = device
        pass

    def reset(self, seed=None) -> torch.Tensor:
        """
        Reset the environment and retur the observation.

        Returns:
            torch.Tensor
        """
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)

        return obs

    def step_async(self, actions: torch.Tensor) -> None:
        """
        Aysnc step in the vectorized environment.

        Args:
            actions (torch.Tensor): Tensor containing actions 
            to be taken in the environment(s).

        Returns:
            None
        """
        actions = actions.cpu()

        if isinstance(actions, torch.LongTensor):
            # squeeze dimensions for discrete actions
            actions = actions.squeeze(1)

        actions = actions.cpu().numpy()
        # print(actions)
        self.venv.step_async(actions)
        pass

    def step_wait(self) -> Tuple[torch.Tensor, \
                                 torch.Tensor, np.ndarray, List]:
        """
        Wait for the step taken with step_async() and return resulting 
        observations, rewards, etc.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()

        return obs, reward, done, info

    def action_masks(self) -> torch.Tensor:
        """
        Retrieves action masks from the vectorized environment, converts them to PyTorch tensor.

        Returns:
            torch.Tensor: Tensor of action masks.
        """
        # Use env_method to dynamically call 'action_masks_wait' on the vectorized environment
        results = self.env_method('action_masks')
        # Convert the list of numpy arrays to a single numpy array before converting to a tensor
        if isinstance(results, list):
            results = np.array(results)  # Combine list of numpy arrays into a single numpy array
        action_masks = torch.from_numpy(results).to(self.device, dtype=torch.uint8)
        return action_masks
