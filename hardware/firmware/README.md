# ESP32 Microcontroller Firmware

This directory contains the physical traffic light actuation firmware for the ESP32 microcontroller.

## Overview
- **Sketch File**: `esp32sketch.ino`
- **Protocol**: Serial (115200 baud)
- **Format**: `E:<state>,N:<state>,S:<state>,W:<state>\n`
  - Where `<state>` is `R` (Red), `Y` (Yellow), or `G` (Green).

## Flashing Instructions
1. Open Arduino IDE or PlatformIO.
2. Ensure the ESP32 board package (by Espressif Systems) is installed.
3. Select your ESP32 board model (e.g., `ESP32 Dev Module`) and the connected serial COM port.
4. Verify GPIO pin assignments defined at the top of `esp32sketch.ino`.
5. Compile and upload the sketch.
6. Verify output in the Serial Monitor at 115200 baud.
