import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network with Layer Normalisation and orthogonal weight initialisation.

    Improvements over the base version:
    - LayerNorm after each hidden layer stabilises activations when Q-values
      span a wide numeric range (common in shaped traffic-reward environments).
    - Orthogonal init with per-layer gain prevents exploding / vanishing
      gradients during early training.
    - Small gain (0.01) on the advantage head keeps initial Q-values near zero,
      avoiding overconfident decisions before any real experience is collected.
    """

    def __init__(self, input_size, output_size, hidden_size=256):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        # Value and Advantage heads
        self.value_head = nn.Linear(hidden_size, 1)
        self.adv_head = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialisation — standard in deep RL for stable gradient flow."""
        relu_gain = nn.init.calculate_gain('relu')
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.orthogonal_(layer.weight, gain=relu_gain)
            nn.init.zeros_(layer.bias)
        # Value head: unit gain — it outputs a scalar state baseline
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
        # Advantage head: tiny gain so Q-values start near-zero,
        # which prevents the agent from committing to a sub-optimal
        # policy before the replay buffer contains real signal.
        nn.init.orthogonal_(self.adv_head.weight, gain=0.01)
        nn.init.zeros_(self.adv_head.bias)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        v = self.value_head(x)
        a = self.adv_head(x)
        # combine: Q = V + (A - mean(A))
        return v + (a - a.mean(dim=1, keepdim=True))
