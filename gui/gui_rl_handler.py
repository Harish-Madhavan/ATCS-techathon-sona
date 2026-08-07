import os
import numpy as np
import torch
from core import config
from rl_model.network import DuelingQNetwork

class RLDecisionMaker:
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                print(f"[RLDecisionMaker] State for '{intersection_name}' unavailable. Cannot size model.")
                return False

            input_size = len(rl_state)
            num_lanes = len(config.TRAFFIC_LIGHT_CONFIG[intersection_name]['phases'])
            duration_options = getattr(config, 'DURATION_OPTIONS', [10, 20, 30])
            num_duration_options = len(duration_options)
            output_size = num_lanes * num_duration_options

            self.rl_model = DuelingQNetwork(input_size, output_size).to(self.device)
            ckpt = torch.load(config.RL_MODEL_PATH, map_location=self.device)
            self.rl_model.load_state_dict(ckpt)
            self.rl_model.eval()
            print(f"[RLDecisionMaker] Loaded '{config.RL_MODEL_PATH}' on {self.device}. (input_size={input_size}, output_size={output_size})")
            return True
        except Exception as e:
            print(f"[RLDecisionMaker Error] Failed to load/initialize model: {e}")
            raise e

    def get_action(self, controller, intersection_name='Intersection1'):
        """
        Queries the model with the current normalized intersection state,
        applying action-space repeat mitigations.
        """
        if not self.rl_model:
            return None

        rl_state = controller.get_rl_state(intersection_name)
        if rl_state is None:
            return None

        try:
            # Normalize inputs
            normalize_max = getattr(config, 'RL_NORMALIZE_MAX', 50.0)
            
            num_lanes = len(config.TRAFFIC_LIGHT_CONFIG[intersection_name]['phases'])
            num_vehicle_types = len(config.TARGET_CLASSES)
            v_end = num_lanes * (num_vehicle_types if num_vehicle_types > 0 else 1)
            w_end = v_end + num_lanes
            
            state_arr = np.array(rl_state, dtype=float)
            state_arr[:v_end] = state_arr[:v_end] / max(1.0, float(normalize_max))
            if len(state_arr) >= w_end:
                state_arr[v_end:w_end] = np.minimum(state_arr[v_end:w_end], 300.0) / 300.0
            
            state_tensor = torch.FloatTensor(state_arr).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.rl_model(state_tensor)
            rl_action = int(torch.argmax(q_values).item())

            # Track recent actions for repeat detection
            self._rl_action_history.append(rl_action)
            if len(self._rl_action_history) > 40:
                self._rl_action_history = self._rl_action_history[-40:]
            
            repeats = 1
            for a in reversed(self._rl_action_history[:-1]):
                if a == self._rl_action_history[-1]:
                    repeats += 1
                else:
                    break

            # Repeat mitigation logic
            max_repeat = getattr(config, 'MAX_CONSECUTIVE_RL_SAME_LANE', 2)
            if repeats > max_repeat:
                q_np = q_values.cpu().numpy()[0]
                duration_options = getattr(config, 'DURATION_OPTIONS', [10, 20, 30])
                num_duration_options = len(duration_options)
                if num_duration_options > 0:
                    repeated_lane = int(rl_action // num_duration_options)
                    mask = np.ones_like(q_np, dtype=bool)
                    for a_idx in range(len(q_np)):
                        if int(a_idx // num_duration_options) == repeated_lane:
                            mask[a_idx] = False
                    if mask.any():
                        masked_q = np.copy(q_np)
                        masked_q[~mask] = -1e9
                        rl_action = int(np.argmax(masked_q))
                        self._rl_action_history[-1] = rl_action
                        print(f"[RLDecisionMaker] Mitigated repeat {repeats} for lane {repeated_lane}. New action: {rl_action}")

            return rl_action

        except Exception as e:
            print(f"[RLDecisionMaker Error] Action selection failed: {e}")
            return None
