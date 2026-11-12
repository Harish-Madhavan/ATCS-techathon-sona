"""Baseline + RL benchmark agents (importable library, no CLI).

Moved here from benchmark.py so tests and tooling can use them without
importing the CLI script. benchmark.py re-exports these for compatibility.
"""
import numpy as np
import torch

from core import config
from core.log import get_logger
from rl_model.contract import (
    build_lane_mask as _lane_mask,
    masked_argmax as _masked_argmax,
)
from rl_model.network import load_q_network

logger = get_logger(__name__)


class RLAgent:
    """Agent that uses a trained Q-network to make decisions."""
    def __init__(self, model_path, input_size, output_size, device, normalize_max=50.0):
        self.model = load_q_network(model_path, input_size, output_size, device)
        self.device = device
        self.normalize_max = max(1.0, float(normalize_max))

    def choose_action(self, state, env=None):
        state_tensor = torch.FloatTensor(np.array(state, dtype=float)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)

        q_np = np.nan_to_num(q_values.cpu().numpy()[0], nan=-1e9, posinf=1e9, neginf=-1e9)
        duration_options = getattr(config, 'DURATION_OPTIONS', [10, 20, 30])
        num_duration_options = len(duration_options) if duration_options else 1

        # Starvation safety mirrors rl_model.inference: force-serve long-waiting lane.
        if env is not None and getattr(config, 'STARVATION_ENABLED', False):
            threshold = float(getattr(config, 'STARVATION_WAIT_S', 180.0))
            waits = getattr(env, 'wait_times', None)
            if waits is not None and len(waits) == env.num_lanes:
                starved = int(np.argmax(waits))
                if float(waits[starved]) > threshold:
                    base = starved * num_duration_options
                    lane_q = q_np[base:base + num_duration_options]
                    return int(base + int(np.argmax(lane_q)))

        # Apply action masking using env if available (shared helper)
        mask = None
        if env is not None:
            max_repeat = getattr(config, 'MAX_CONSECUTIVE_RL_SAME_LANE', 2)
            if env.consecutive_lane_count >= max_repeat:
                mask = _lane_mask(len(q_np) // num_duration_options,
                                  num_duration_options, [env.current_lane])
        if mask is not None:
            return _masked_argmax(q_np, mask)

        return _masked_argmax(q_np, np.ones_like(q_np, dtype=bool))


class StaticAgent:
    """Agent that cycles through lanes with a fixed green light duration."""
    def __init__(self, num_lanes, duration_options, fixed_duration=30, normalize_max=50.0):
        self.num_lanes = num_lanes
        self.duration_options = duration_options
        self.num_duration_options = len(duration_options)
        self.fixed_duration = fixed_duration
        self.normalize_max = max(1.0, float(normalize_max))
        try:
            self.duration_index = self.duration_options.index(self.fixed_duration)
        except ValueError:
            self.duration_index = min(range(len(self.duration_options)), key=lambda i: abs(self.duration_options[i] - self.fixed_duration))
        self.current_lane = 0

    def reset(self):
        """Reset lane pointer so each benchmarked episode starts from lane 0."""
        self.current_lane = 0

    def choose_action(self, state, env=None):
        action = self.current_lane * self.num_duration_options + self.duration_index
        self.current_lane = (self.current_lane + 1) % self.num_lanes
        return action


class GreedyAgent:
    """Agent that greedily selects the lane with the highest queue weight, prioritizing emergency vehicles."""
    def __init__(self, num_lanes, duration_options, fixed_duration=20, normalize_max=50.0):
        self.num_lanes = num_lanes
        self.duration_options = duration_options
        self.num_duration_options = len(duration_options)
        self.normalize_max = max(1.0, float(normalize_max))
        try:
            self.duration_index = self.duration_options.index(fixed_duration)
        except ValueError:
            self.duration_index = min(range(len(self.duration_options)), key=lambda i: abs(self.duration_options[i] - fixed_duration))

    def choose_action(self, state, env=None):
        if env is not None:
            # Check for ambulances
            for ln in range(self.num_lanes):
                if env.ambulance_presence[ln] > 0:
                    return ln * self.num_duration_options + self.duration_index

            # Greedy select lane with highest pre-demand
            demands = np.sum(env.state_vehicle_counts * env.vehicle_weights, axis=1)
            lane = int(np.argmax(demands))
        else:
            lane = 0
        return lane * self.num_duration_options + self.duration_index
