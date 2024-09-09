from typing import Tuple

import torch.nn as nn
import torch

from rl_squared.utils.torch_utils import init_gru, init_lstm



class LSTM(nn.Module):
    def __init__(self, input_size: int, recurrent_state_size: int):
        """
        Stateful actor for a discrete action space.

        Args:
            input_size (int): State dimensions for the environment.
            recurrent_state_size (int): Size of the recurrent state.
        """
        nn.Module.__init__(self)

        # self._lstm = init_gru(input_size, recurrent_state_size)
        self._lstm = init_lstm(input_size, recurrent_state_size)

    def forward(
        self,
        x: torch.Tensor,
        recurrent_states: torch.Tensor,
        recurrent_state_masks: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the LSTM.

        Args:
            x (torch.Tensor): Input tensor.
            recurrent_states (torch.Tensor): Combined hidden and cell state from the previous step.
            recurrent_state_masks (torch.Tensor): Masks to be applied to the recurrent states.
            device (torch.device): Device to perform computations on.

        Returns:
            Tuple containing output tensor from the LSTM and the new combined hidden and cell state.
        """

        print(f"original x.shape: {x.shape}")
        print(f"original recurrent_states.shape: {recurrent_states.shape}")
        # print(recurrent_state_masks.shape)

        x = x.unsqueeze(0)
        print(f"x after unsqueezing: {x.shape}")
        recurrent_states = recurrent_states.unsqueeze(0)
        print(f"recurrent_states.shape {recurrent_states.shape}")

        # Split the combined recurrent state into hidden state (h) and cell state (c)
        seq_length, batch_size, _ = recurrent_states.shape
        hidden_dim = recurrent_states.shape[2] // 2  # Assuming the hidden and cell states are concatenated along the last dimension
        # print(hidden_dim)
        h, c = torch.split(recurrent_states, hidden_dim, dim=2)

        print(f"x shape: {x.shape}")
        print(f"h shape: {h.shape}")
        print(f"c shape: {c.shape}")


        # Ensure h and c are contiguous in memory
        h = h.contiguous()
        c = c.contiguous()

        # Pass x, h, c to self._lstm and perform the LSTM forward pass
        output, (new_h, new_c) = self._lstm(x, (h, c))

        # Combine new_h and new_c into new recurrent states
        new_recurrent_states = torch.cat((new_h, new_c), dim=2)

        return output.squeeze(0), new_recurrent_states.squeeze(0)



# Example of usage
if __name__ == "__main__":
    input_size = 10
    recurrent_state_size = 20
    batch_size = 3

    lstm_module = LSTM(input_size, recurrent_state_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_module.to(device)

    # Example with batch_size = 1 for simplicity
    x = torch.randn(batch_size, input_size).to(device)
    recurrent_states = torch.randn(batch_size, recurrent_state_size).to(device)  # Combined size for hidden and cell
    # recurrent_state_masks = torch.ones_like(recurrent_states).to(device)
    recurrent_state_masks = None
    
    outputs, new_states = lstm_module(x, recurrent_states, recurrent_state_masks, device)
    print("Outputs shape:", outputs.shape)
    print("New states shape:", new_states.shape)