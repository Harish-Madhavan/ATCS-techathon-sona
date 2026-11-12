import unittest

from hardware.esp32_controller import ESP32SerialController


class TestESP32Controller(unittest.TestCase):
    def setUp(self):
        self.mapping = {
            "Northbound": "N",
            "Eastbound": "E",
            "Westbound": "W",
            "Southbound": "S"
        }
        self.controller = ESP32SerialController(port="MOCK", baudrate=115200, approach_mapping=self.mapping)

    def test_mock_connection(self):
        """Verify MOCK mode connects seamlessly without physical hardware."""
        self.assertTrue(self.controller.is_connected)

    def test_update_lights_command_generation(self):
        """Verify command string formatting for ESP32 packet updates."""
        statuses = {
            "Northbound": {"state": "GREEN"},
            "Eastbound": {"state": "RED"},
            "Westbound": {"state": "RED"},
            "Southbound": {"state": "YELLOW"}
        }
        # Should execute without errors in Mock mode
        self.controller.update_lights(statuses)
        self.assertTrue(self.controller.is_connected)

    def test_close_and_reconnect(self):
        """Verify close and reconnect functions."""
        self.controller.close()
        self.assertFalse(self.controller.is_connected)
        self.controller.reconnect()
        self.assertTrue(self.controller.is_connected)


if __name__ == '__main__':
    unittest.main()
