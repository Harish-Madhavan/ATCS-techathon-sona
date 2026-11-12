import os
import torch
import torch.nn as nn

MODEL_VERSION = 2


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
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_size = int(hidden_size)
        self.model_version = MODEL_VERSION
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

    @torch.no_grad()
    def predict(self, state_vector, device=None):
        """Numpy in -> numpy Q out. Never returns NaN (sanitised)."""
        import numpy as np
        dev = device or next(self.parameters()).device
        t = torch.as_tensor(np.asarray(state_vector, dtype=float),
                            dtype=torch.float32, device=dev).unsqueeze(0)
        q = self.forward(t).cpu().numpy()[0]
        return np.nan_to_num(q, nan=-1e9, posinf=1e9, neginf=-1e9)

    def save_checkpoint(self, path, extra=None):
        """Save state_dict + metadata sidecar-compatibly (old loaders keep working)."""
        payload = {
            "version": self.model_version,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "state_dict": self.state_dict(),
        }
        if extra:
            payload["extra"] = dict(extra)
        torch.save(payload, path)
        # Plain state_dict copy for legacy loaders that expect raw weights.
        legacy = os.path.splitext(path)[0] + ".legacy.pth"
        try:
            torch.save(self.state_dict(), legacy)
        except Exception:
            pass
        return path


def load_q_network(path, input_size, output_size, device, hidden_size=256):
    """Load with clear dim-mismatch errors; accepts legacy raw state_dicts.

    Raises RuntimeError with expected-vs-found dims instead of cryptic matmul errors.
    """
    from core.log import get_logger
    log = get_logger(__name__)
    raw = torch.load(path, map_location=device, weights_only=False)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if isinstance(raw, dict) and "version" in raw:
        log.debug("Loading RL checkpoint v%s (%s)", raw.get("version"), path)
    # Infer checkpoint dims from adv head for a helpful error.
    found_out = None
    try:
        for k in ("adv_head.weight", "adv_head.bias"):
            if isinstance(state, dict) and k in state and hasattr(state[k], "shape"):
                found_out = int(state[k].shape[0])
                break
    except Exception:
        pass
    found_in = None
    try:
        if isinstance(state, dict) and "fc1.weight" in state and hasattr(state["fc1.weight"], "shape"):
            found_in = int(state["fc1.weight"].shape[1])
    except Exception:
        pass
    if found_out is not None and found_out != int(output_size):
        raise RuntimeError(
            f"RL model output mismatch: checkpoint has {found_out}, "
            f"config needs {output_size}. Retrain or fix DURATION_OPTIONS."
        )
    if found_in is not None and found_in != int(input_size):
        raise RuntimeError(
            f"RL model input mismatch: checkpoint has {found_in}, "
            f"config needs {input_size}. Retrain or fix TARGET_CLASSES."
        )
    model = DuelingQNetwork(input_size, output_size, hidden_size=hidden_size).to(device)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(f"RL checkpoint incompatible with current config: {e}") from e
    model.eval()
    return model
