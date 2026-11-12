import unittest
import numpy as np
import torch
import tempfile
import os

from rl_model.contract import (
    action_dim, state_dim, decode_action, validate_action,
    build_lane_mask, masked_argmax, greedy_lane_by_demand,
)
from rl_model.network import DuelingQNetwork, load_q_network, MODEL_VERSION
from rl_model.environment import TrafficEnv


class TestContract(unittest.TestCase):
    def test_dims_match_runtime(self):
        self.assertEqual(state_dim(4, 6), 36)
        self.assertEqual(action_dim(4, 12), 48)

    def test_decode_roundtrip(self):
        for lane in range(4):
            for dur in range(12):
                a = lane * 12 + dur
                self.assertEqual(decode_action(a, 12), (lane, dur))
        with self.assertRaises(ValueError):
            decode_action(-1, 12)
        with self.assertRaises(ValueError):
            validate_action(48, 4, 12)

    def test_lane_mask(self):
        m = build_lane_mask(4, 3, [1])
        self.assertEqual(len(m), 12)
        self.assertFalse(m[3:6].any())
        self.assertTrue(m[0:3].all())

    def test_masked_argmax_nan_and_all_masked(self):
        q = np.array([np.nan, 5.0, 1.0, 2.0])
        m = np.array([True, True, True, True])
        self.assertEqual(masked_argmax(q, m), 1)
        m_all = np.array([False, False, False, False])
        self.assertEqual(masked_argmax(q, m_all, fallback_greedy=2), 2)
        self.assertEqual(masked_argmax(q, m_all), 0)

    def test_greedy_fallback_skips_blocked(self):
        self.assertEqual(greedy_lane_by_demand([1.0, 9.0, 2.0], [1]), 2)


class TestNetworkIO(unittest.TestCase):
    def test_version_attrs_and_predict_no_nan(self):
        net = DuelingQNetwork(36, 48)
        self.assertEqual(net.model_version, MODEL_VERSION)
        q = net.predict(np.zeros(36))
        self.assertEqual(q.shape, (48,))
        self.assertFalse(np.isnan(q).any())

    def test_load_mismatch_error_is_clear(self):
        net = DuelingQNetwork(36, 48)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pth")
            torch.save(net.state_dict(), p)
            with self.assertRaisesRegex(RuntimeError, "output mismatch"):
                load_q_network(p, 36, 24, torch.device("cpu"))
            with self.assertRaisesRegex(RuntimeError, "input mismatch"):
                load_q_network(p, 12, 48, torch.device("cpu"))

    def test_load_new_metadata_format(self):
        net = DuelingQNetwork(12, 8)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m2.pth")
            net.save_checkpoint(p)
            loaded = load_q_network(p, 12, 8, torch.device("cpu"))
            self.assertEqual(loaded.output_size, 8)


class TestEnvGuards(unittest.TestCase):
    def test_invalid_action_rejected(self):
        env = TrafficEnv(num_lanes=4, vehicle_type_names=["a", "b"],
                         vehicle_weights=[1.0, 2.0], duration_options=[10, 20])
        env.reset()
        with self.assertRaises(ValueError):
            env.step(999)

    def test_starved_lane_helper(self):
        env = TrafficEnv(num_lanes=4, vehicle_type_names=["a"],
                         vehicle_weights=[1.0], duration_options=[10])
        env.reset()
        env.wait_times[:] = [10.0, 200.0, 5.0, 0.0]
        self.assertEqual(env.get_starved_lane(180.0), 1)
        self.assertIsNone(env.get_starved_lane(500.0))

    def test_seed_reproducible_reset(self):
        kwargs = dict(num_lanes=4, vehicle_type_names=["a", "b"],
                      vehicle_weights=[1.0, 2.0], duration_options=[10, 20])
        e1, e2 = TrafficEnv(**kwargs), TrafficEnv(**kwargs)
        e1.seed(123)
        e2.seed(123)
        np.testing.assert_array_equal(e1.reset(), e2.reset())


if __name__ == "__main__":
    unittest.main()
