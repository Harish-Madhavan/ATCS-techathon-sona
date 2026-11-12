"""Single source of truth for the RL state/action contract.

State layout (matches TrafficLightController.get_rl_state and TrafficEnv._get_state):
    [counts(lanes*types) | wait(lanes) | ambulance(lanes) | one_hot(lanes)]

Action encoding:
    action = lane_index * num_durations + duration_index
"""
import numpy as np


def state_dim(num_lanes, num_vehicle_types):
    n_types = num_vehicle_types if num_vehicle_types and num_vehicle_types > 0 else 1
    return num_lanes * n_types + num_lanes * 3


def action_dim(num_lanes, num_durations):
    return int(num_lanes) * int(num_durations)


def decode_action(action, num_durations):
    """Return (lane_index, duration_index); raises ValueError on out-of-range."""
    num_durations = int(num_durations)
    if num_durations <= 0:
        raise ValueError("num_durations must be > 0")
    action = int(action)
    if action < 0:
        raise ValueError(f"Negative action {action}")
    return action // num_durations, action % num_durations


def validate_action(action, num_lanes, num_durations):
    lane, dur = decode_action(action, num_durations)
    if not (0 <= lane < num_lanes):
        raise ValueError(f"Lane {lane} out of range 0..{num_lanes - 1}")
    if not (0 <= dur < num_durations):
        raise ValueError(f"Duration idx {dur} out of range 0..{num_durations - 1}")
    return True


def build_lane_mask(num_lanes, num_durations, blocked_lanes=()):
    """Boolean mask (True=allowed) over the flat action space."""
    blocked = set(int(b) for b in (blocked_lanes or ()))
    mask = np.ones(action_dim(num_lanes, num_durations), dtype=bool)
    for lane in blocked:
        if 0 <= lane < num_lanes:
            base = lane * num_durations
            mask[base:base + num_durations] = False
    return mask


def masked_argmax(q_values, mask, fallback_greedy=None):
    """Argmax over allowed actions; falls back gracefully when all masked/NaN."""
    q = np.asarray(q_values, dtype=float).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    if q.shape[0] != m.shape[0]:
        raise ValueError(f"q ({q.shape}) and mask ({m.shape}) size mismatch")
    safe = np.where(m, np.nan_to_num(q, nan=-1e9, posinf=1e9, neginf=-1e9), -np.inf)
    if np.all(np.isneginf(safe)):
        if fallback_greedy is not None:
            return int(fallback_greedy)
        # Least-bad default: action 0 rather than crashing the FSM.
        return 0
    return int(np.argmax(safe))


def greedy_lane_by_demand(counts_by_lane_weight, blocked_lanes=()):
    """Pure-numpy fallback: lane with max weighted demand, skipping blocked."""
    demands = np.asarray(list(counts_by_lane_weight), dtype=float)
    blocked = set(int(b) for b in (blocked_lanes or ()))
    best, best_val = 0, -1.0
    found = False
    for i, v in enumerate(demands):
        if i in blocked:
            continue
        if (not found) or v > best_val:
            best, best_val, found = int(i), float(v), True
    return best if found else 0
