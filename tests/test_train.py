import unittest
import tempfile
import shutil
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from rl_model.train import train, main


class TestRLTraining(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_train_single_episode_cpu(self):
        output_model = os.path.join(self.test_dir, "test_rl_model.pth")
        ckpts_dir = os.path.join(self.test_dir, "ckpts")

        model = train(
            num_lanes=4,
            vehicle_type_names=["car", "truck"],
            vehicle_weights=[1.0, 2.0],
            duration_options=[10, 20],
            num_episodes=1,
            batch_size=8,
            replay_buffer_size=100,
            learning_rate=1e-3,
            device_str="cpu",
            seed=123,
            use_time_varying_arrival_rate=False,
            checkpoint_dir=ckpts_dir,
            model_output_path=output_model,
        )

        self.assertIsNotNone(model)
        self.assertTrue(os.path.exists(output_model))
        # Ensure saved model can be loaded back
        state_dict = torch.load(output_model, map_location="cpu", weights_only=True)
        self.assertIn("fc1.weight", state_dict)

    def test_train_cli_smoke(self):
        output_model = os.path.join(self.test_dir, "cli_test_model.pth")
        ret = main(["--episodes", "1", "--device", "cpu", "--static-rates", "--output", output_model])
        self.assertEqual(ret, 0)
        self.assertTrue(os.path.exists(output_model))


if __name__ == '__main__':
    unittest.main()
