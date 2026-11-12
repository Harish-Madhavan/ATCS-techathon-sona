"""ATCS Microcontroller & Hardware Bridge.

Provides:
- Physical ESP32 serial communication driver and mock interface (`ESP32SerialController`)
- Arduino C++ firmware sketch in `hardware/firmware/`
"""
from hardware.esp32_controller import ESP32SerialController

__all__ = [
    "ESP32SerialController",
]
