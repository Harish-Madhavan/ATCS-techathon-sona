import unittest
import numpy as np
import torch

from rl_model.device import get_best_device, get_best_device_str
from rl_model.environment import TrafficEnv
from hardware.esp32_controller import ESP32SerialController
from core.log import get_logger


class TestDeviceHelper(unittest.TestCase):
    def test_returns_valid_device(self):
        d = get_best_device()
        self.assertIn(str(d), ("cpu", "cuda", "xpu"))
        self.assertIn(get_best_device_str(), ("cpu", "cuda", "xpu"))


class TestLogging(unittest.TestCase):
    def test_get_logger(self):
        log = get_logger("test")
        self.assertTrue(hasattr(log, "info"))
        # Core FSM should log quietly without printing to stdout
        from core.traffic_logic import TrafficLightController
        from core import config
        ctrl = TrafficLightController(
            config.TRAFFIC_LIGHT_CONFIG,
            config.VEHICLE_TYPE_WEIGHTS,
            config.DEFAULT_VEHICLE_WEIGHT,
        )
        st = ctrl.intersections["Intersection1"]
        st["last_update_time"] = 0.0
        with self.assertLogs("core.intersection_state", level="INFO"):
            ctrl.update_state(1.5)


class TestESP32BuildCommand(unittest.TestCase):
    def test_deterministic_order(self):
        c = ESP32SerialController(
            port="MOCK", baudrate=115200,
            approach_mapping={"Northbound": "N", "Eastbound": "E"},
        )
        cmd = c.build_command({
            "Northbound": {"state": "GREEN"},
            "Eastbound": {"state": "RED"},
            "Unknown": {"state": "GREEN"},
        })
        self.assertEqual(cmd, "E:R,N:G\n")
        c.close()


class TestStarvationOverride(unittest.TestCase):
    def test_forces_long_wait_lane(self):
        from core import config
        from core.traffic_logic import TrafficLightController
        from rl_model.inference import RLDecisionMaker

        ctrl = TrafficLightController(
            config.TRAFFIC_LIGHT_CONFIG,
            config.VEHICLE_TYPE_WEIGHTS,
            config.DEFAULT_VEHICLE_WEIGHT,
        )
        state = ctrl.intersections["Intersection1"]
        # Simulate lane Southbound starved
        state["approach_wait_time"]["Southbound"] = float(config.STARVATION_WAIT_S) + 50.0
        state["approach_wait_time"]["Northbound"] = 0.0

        mk = RLDecisionMaker(device=torch.device("cpu"))
        # Stub a model that would otherwise pick lane 0
        import torch.nn as nn
        from rl_model.network import DuelingQNetwork
        rl_state = ctrl.get_rl_state("Intersection1")
        mk.rl_model = DuelingQNetwork(len(rl_state), 4 * len(config.DURATION_OPTIONS)).to("cpu")
        mk.rl_model.eval()
        # Zero weights -> argmax 0, but starvation must force Southbound lane
        with torch.no_grad():
            for p in mk.rl_model.parameters():
                p.zero_()
        action = mk.get_action(ctrl, "Intersection1")
        num_dur = len(config.DURATION_OPTIONS)
        # Southbound is phase index 3 in default order
        self.assertEqual(action // num_dur, 3)


class TestCanonicalNormalization(unittest.TestCase):
    def test_matches_env(self):
        env = TrafficEnv(num_lanes=4, vehicle_type_names=["a", "b"],
                         vehicle_weights=[1.0, 2.0], duration_options=[10, 20])
        raw = env.reset()
        a = env.normalize_state(raw, 50.0)
        b = TrafficEnv.normalize_state_vector(raw, 4, 2, 50.0)
        np.testing.assert_allclose(a, b)


if __name__ == "__main__":
    unittest.main()
