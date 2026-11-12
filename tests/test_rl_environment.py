import unittest
import numpy as np
import torch

from rl_model.environment import TrafficEnv
from rl_model.network import DuelingQNetwork


class TestRLEnvironment(unittest.TestCase):
    def setUp(self):
        self.num_lanes = 4
        self.vehicle_types = ['car', 'bus', 'truck']
        self.weights = [1.0, 3.0, 2.0]
        self.duration_options = [10, 20, 30]
        self.env = TrafficEnv(
            num_lanes=self.num_lanes,
            vehicle_type_names=self.vehicle_types,
            vehicle_weights=self.weights,
            duration_options=self.duration_options,
            arrival_rate_per_lane=[0.2] * self.num_lanes,
            max_queue=50,
            total_steps=50
        )

    def test_environment_reset(self):
        """Verify environment reset initializes states correctly."""
        state = self.env.reset()
        self.assertIsInstance(state, np.ndarray)
        expected_len = (self.num_lanes * len(self.vehicle_types)) + self.num_lanes + self.num_lanes + self.num_lanes
        self.assertEqual(len(state), expected_len)
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.current_time, 0.0)

    def test_environment_step(self):
        """Verify environment step advances time and updates queues."""
        self.env.reset()
        action = 0  # Lane 0, 10s
        next_state, reward, done, info = self.env.step(action)
        self.assertEqual(self.env.current_time, 10.0)
        self.assertEqual(self.env.current_step, 1)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(info, dict)
        self.assertIn("vehicles_passed", info)
        self.assertIn("cleared_value", info)

    def test_state_normalization(self):
        """Verify log1p wait time scaling and queue normalization bounds."""
        state = self.env.reset()
        # Set large wait times to test log scaling
        self.env.wait_times[0] = 300.0
        raw_state = self.env._get_state()
        norm_state = self.env.normalize_state(raw_state, normalize_max=50.0)

        self.assertEqual(len(raw_state), len(norm_state))
        # Normalized values should be within reasonable bounded ranges
        self.assertTrue(np.all(norm_state >= 0.0))
        # Wait time at 300s normalized with log1p(300)/log1p(300) should be approx 1.0
        v_end = self.num_lanes * len(self.vehicle_types)
        self.assertAlmostEqual(norm_state[v_end], 1.0, places=2)

    def test_dueling_q_network_forward(self):
        """Verify DuelingQNetwork forward pass with LayerNorm."""
        input_size = (self.num_lanes * len(self.vehicle_types)) + 3 * self.num_lanes
        output_size = self.num_lanes * len(self.duration_options)

        model = DuelingQNetwork(input_size, output_size)
        x = torch.randn(2, input_size)
        q_values = model(x)

        self.assertEqual(q_values.shape, (2, output_size))
        self.assertFalse(torch.isnan(q_values).any())


if __name__ == '__main__':
    unittest.main()
