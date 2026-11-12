import sys
import os
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_model.agents import RLAgent
from rl_model.environment import TrafficEnv
from core import config
from core.config import TARGET_CLASSES, VEHICLE_TYPE_WEIGHTS, DURATION_OPTIONS


class TestQInference(unittest.TestCase):
    def test_q_values_inference(self):
        num_lanes = 4
        vehicle_type_names = TARGET_CLASSES
        vt_weights = [float(VEHICLE_TYPE_WEIGHTS.get(vt, 1.0)) for vt in vehicle_type_names]
        duration_options = DURATION_OPTIONS
        v_end = num_lanes * len(vehicle_type_names)
        input_size = v_end + 3 * num_lanes
        output_size = num_lanes * len(duration_options)

        device = torch.device("cpu")
        env = TrafficEnv(num_lanes=num_lanes, vehicle_type_names=vehicle_type_names, vehicle_weights=vt_weights, duration_options=duration_options)
        
        if os.path.exists(config.RL_MODEL_PATH):
            agent = RLAgent(config.RL_MODEL_PATH, input_size, output_size, device)
            state = env.reset()
            normalized_state = env.normalize_state(state, agent.normalize_max)
            state_tensor = torch.FloatTensor(np.array(normalized_state, dtype=float)).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent.model(state_tensor)

            self.assertEqual(q_values.shape, (1, output_size))
            action = int(torch.argmax(q_values).item())
            self.assertTrue(0 <= action < output_size)


if __name__ == '__main__':
    unittest.main()
