import os
import time
import numpy as np
import torch
from core import config
from core.log import get_logger
from rl_model.network import load_q_network

from rl_model.environment import TrafficEnv
from rl_model.device import get_best_device
from rl_model.contract import (
    action_dim as _action_dim,
    build_lane_mask as _lane_mask,
    decode_action as _decode,
    masked_argmax as _masked_argmax,
    state_dim as _state_dim,
)

logger = get_logger(__name__)

class RLDecisionMaker:
    def __init__(self, device=None):
        if device is None:
            device = get_best_device()
        self.device = device
        self.rl_model = None
        self._rl_action_history = []

    def load_model(self, controller, intersection_name='Intersection1'):
        """
        Dynamically initializes the model shape based on the controller's state length,
        then loads the model state dictionary from config.RL_MODEL_PATH.
        """
        if not config.RL_MODEL_ENABLED:
            return False

        if not os.path.exists(config.RL_MODEL_PATH):
            print(f"[RLDecisionMaker] RL model file not found at '{config.RL_MODEL_PATH}'.")
            return False

        try:
            rl_state = controller.get_rl_state(intersection_name)
            if rl_state is None:
                logger.warning("State for '%s' unavailable. Cannot size model.", intersection_name)
                return False

            num_lanes = len(config.TRAFFIC_LIGHT_CONFIG[intersection_name]['phases'])
            num_types = len(getattr(config, 'TARGET_CLASSES', []))
            duration_options = getattr(config, 'DURATION_OPTIONS', [10, 20, 30])
            input_size = _state_dim(num_lanes, num_types)
            output_size = _action_dim(num_lanes, len(duration_options))
            if len(rl_state) != input_size:
                logger.warning(
                    "Live state len %d != contract %d; loading by live len for compat.",
                    len(rl_state), input_size,
                )
                input_size = len(rl_state)

            self.rl_model = load_q_network(
                config.RL_MODEL_PATH, input_size, output_size, self.device
            )
            self._input_size = input_size
            self._output_size = output_size
            logger.info("Loaded '%s' on %s (in=%d, out=%d).",
                        config.RL_MODEL_PATH, self.device, input_size, output_size)
            return True
        except Exception as e:
            logger.error("Failed to load/initialize model: %s", e)
            raise e

    def get_action(self, controller, intersection_name='Intersection1'):
        """
        Queries the model with the current normalized intersection state,
        applying action-space repeat mitigations and manual red override masking.
        """
        if not self.rl_model:
            return None

        rl_state = controller.get_rl_state(intersection_name)
        if rl_state is None:
            return None

        try:
            int_state = controller.intersections.get(intersection_name, {})
            phase_order = int_state.get('rl_phase_order', int_state.get('phases', []))
            phases_cfg = int_state.get('config', {}).get('phases', {})
            manual_overrides = int_state.get('manual_override_red', {})

            num_lanes = len(config.TRAFFIC_LIGHT_CONFIG[intersection_name]['phases'])
            num_vehicle_types = len(config.TARGET_CLASSES)
            normalize_max = getattr(config, 'RL_NORMALIZE_MAX', 50.0)

            # Canonical normalization
            state_arr = TrafficEnv.normalize_state_vector(
                rl_state, num_lanes, num_vehicle_types, normalize_max
            )
            state_tensor = torch.FloatTensor(state_arr).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.rl_model(state_tensor)

            t0 = time.perf_counter()
            q_np = np.nan_to_num(q_values.cpu().numpy()[0],
                                 nan=-1e9, posinf=1e9, neginf=-1e9)
            duration_options = getattr(config, 'DURATION_OPTIONS', [10, 20, 30])
            num_duration_options = len(duration_options) if duration_options else 1
            max_repeat = getattr(config, 'MAX_CONSECUTIVE_RL_SAME_LANE', 2)

            # Mask 1: Manually overridden red lanes (shared helper)
            blocked = [
                lane_idx for lane_idx, phase_key in enumerate(phase_order)
                if (phases_cfg.get(phase_key, [None])[0]
                    and manual_overrides.get(phases_cfg.get(phase_key, [None])[0], False))
            ]
            mask = _lane_mask(num_lanes, num_duration_options, blocked)

            # Mask 2: Consecutive repeat limit.
            # Prefer the FSM's authoritative counters (single source of truth)
            # so a fallback lane swap inside IntersectionState can't desync us.
            last_lane = None
            repeats = 0
            last_phase_key = int_state.get('last_rl_chosen_lane')
            if last_phase_key in phase_order:
                last_lane = phase_order.index(last_phase_key)
                repeats = int(int_state.get('consecutive_rl_same_lane', 0) or 0)
            elif self._rl_action_history:
                last_act = self._rl_action_history[-1]
                last_lane = int(last_act // num_duration_options)
                repeats = 1
                for a in reversed(self._rl_action_history[:-1]):
                    if int(a // num_duration_options) == last_lane:
                        repeats += 1
                    else:
                        break

            if last_lane is not None and repeats >= max_repeat:
                base = last_lane * num_duration_options
                mask[base:base + num_duration_options] = False

            # Starvation safety: force-serve a lane waiting too long.
            # Overrides the repeat limiter but never manual red / emergency
            # (emergency is handled first inside the FSM).
            if getattr(config, 'STARVATION_ENABLED', False):
                threshold = float(getattr(config, 'STARVATION_WAIT_S', 180.0))
                wait_map = int_state.get('approach_wait_time', {}) or {}
                best_lane = None
                best_wait = threshold
                for lane_idx, phase_key in enumerate(phase_order):
                    appr = phases_cfg.get(phase_key, [None])[0]
                    if not appr or manual_overrides.get(appr, False):
                        continue
                    w = float(wait_map.get(appr, 0.0) or 0.0)
                    if w > best_wait:
                        best_wait = w
                        best_lane = lane_idx
                if best_lane is not None:
                    lane_q = q_np[
                        best_lane * num_duration_options:(best_lane + 1) * num_duration_options
                    ]
                    # Respect manual mask: if whole lane masked, skip starvation.
                    lane_mask = mask[
                        best_lane * num_duration_options:(best_lane + 1) * num_duration_options
                    ]
                    if lane_mask.any():
                        lane_q = np.where(lane_mask, lane_q, -1e9)
                        rl_action = int(best_lane * num_duration_options + int(np.argmax(lane_q)))
                        self._rl_action_history.append(rl_action)
                        if len(self._rl_action_history) > 40:
                            self._rl_action_history = self._rl_action_history[-40:]
                        return rl_action

            # If all actions were masked, fallback to allow unblocked lanes
            if not mask.any():
                mask = _lane_mask(num_lanes, num_duration_options, blocked)

            # Greedy demand fallback if Q is flat/NaN: pick busiest unblocked lane.
            fallback = None
            try:
                counts = []
                for phase_key in phase_order:
                    appr = phases_cfg.get(phase_key, [None])[0]
                    per_type = (int_state.get('approach_counts_by_type', {}) or {}).get(appr, {})
                    wmap = getattr(config, 'VEHICLE_TYPE_WEIGHTS', {})
                    tot = sum(float(per_type.get(k, 0)) * float(wmap.get(k, 1.0)) for k in per_type) if per_type else 0.0
                    counts.append(tot)
                from rl_model.contract import greedy_lane_by_demand as _greedy
                glane = _greedy(counts, blocked)
                fallback = int(glane * num_duration_options + int(np.argmax(
                    np.where(mask[glane * num_duration_options:(glane + 1) * num_duration_options],
                             q_np[glane * num_duration_options:(glane + 1) * num_duration_options], -1e9))))
            except Exception:
                fallback = None

            rl_action = _masked_argmax(q_np, mask, fallback_greedy=fallback)

            # Track action history
            self._rl_action_history.append(rl_action)
            if len(self._rl_action_history) > 40:
                self._rl_action_history = self._rl_action_history[-40:]

            dt_ms = (time.perf_counter() - t0) * 1000.0
            if dt_ms > 50.0:
                logger.warning("RL inference slow: %.1fms", dt_ms)
            else:
                logger.debug("RL inference %.1fms -> action %d", dt_ms, rl_action)
            return rl_action

        except Exception as e:
            logger.exception("Action selection failed: %s", e)
            return None
