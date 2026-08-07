# Codebase Analysis Report: Adaptive Traffic Control System (ATCS)

## Abstract
Traditional traffic signals use fixed timers, causing severe delays and idle emissions. The **Adaptive Traffic Control System (ATCS)** is an autonomous solution combining computer vision (CV) and Deep Reinforcement Learning (DRL) for dynamic signal timing. Using **YOLOv11** for vehicle detection and a **Dueling Double Deep Q-Network (D3QN)** for decision-making, ATCS adjusts phase durations in real time based on queue weights. Hardware-in-the-loop is achieved via an **ESP32 microcontroller**. Benchmarking shows a **35.2% reduction in traffic pressure** and a **59.1% faster response for emergency vehicles** over static systems.

---

## 1. Introduction
Urban congestion escalates delays and environmental waste. Pre-programmed traffic cycles cannot adapt to dynamic fluctuations. The ATCS replaces static systems with a closed-loop framework. By analyzing live camera feeds, it assesses lane demand, computes optimal green-light phase lengths (5s to 60s), and acts physically through microcontrollers, overseen by a desktop dashboard.

---

## 2. Problem Statement
Traditional traffic management suffers from:
- **Rigid Timers**: Inefficient green-light usage on empty roads while adjacent lanes back up.
- **Inadequate Sensors**: Inductive loops are expensive, wear out, and cannot classify vehicle sizes.
- **No Emergency Priority**: Lack of dynamic perception traps emergency vehicles in queue.
- **CPU Bottlenecks**: Linear integration of heavy neural networks and UI loops creates dangerous execution latency.

---

## 3. Objectives
- **Dynamic Perception**: Track lane queues using YOLOv11m-seg.
- **Transit Weighting**: Prioritize heavier vehicles (buses/trucks) over light passenger cars.
- **Emergency Override**: Detect and prioritize ambulances instantly.
- **Actuation Validation**: Communicate phase transitions to an ESP32 serial bridge.
- **Decoupled Processing**: Implement multiprocessing to guarantee real-time execution.
- **Dashboard Monitoring**: Provide operators with traffic statistics and manual override buttons.

---

## 4. Existing Systems
- **Fixed-Cycle Timers**: Low cost but zero real-time responsiveness.
- **SCATS / SCOOT**: Adaptive but demand expensive roadwork for sensors (induction loops/radar) and lack vehicle type classification.
- **Standard Motion Cameras**: Weather-sensitive visual triggers with no capacity to prioritize transit or identify emergency vehicles.

---

## 5. Proposed System
ATCS operates as a decoupled, multi-processed system. First, the perception system captures camera feeds, detects vehicles, and performs point-in-polygon checks for active lanes. Next, the cognition engine evaluates a 36-dimensional state vector (representing lane counts, wait times, ambulance flags, and active lanes) through a trained D3QN network. Finally, the selected phase is sent via serial communication to the ESP32 microcontroller, which actuates the physical traffic LEDs, while the desktop dashboard updates in real-time.

---

## 6. System Modules
- **Perception (`vision/`)**: Handles ROI definitions in [polygon_utils.py](file:///E:/Git%20Repo/ATCS-techathon-sona/vision/polygon_utils.py) and YOLO inference/counting in [video_processor.py](file:///E:/Git%20Repo/ATCS-techathon-sona/vision/video_processor.py).
- **Decision Engine (`rl_model/`)**: Implements the D3QN network in [network.py](file:///E:/Git%20Repo/ATCS-techathon-sona/rl_model/network.py), the custom environment in [environment.py](file:///E:/Git%20Repo/ATCS-techathon-sona/rl_model/environment.py), and the DDQN training code in [train.py](file:///E:/Git%20Repo/ATCS-techathon-sona/train.py).
- **FSM Logic (`core/`)**: Houses the safety FSM logic in [intersection_state.py](file:///E:/Git%20Repo/ATCS-techathon-sona/core/intersection_state.py) and connects the DRL outputs to the controller in [traffic_logic.py](file:///E:/Git%20Repo/ATCS-techathon-sona/core/traffic_logic.py).
- **Hardware (`hardware/`)**: Drives physical LEDs via serial in [esp32_controller.py](file:///E:/Git%20Repo/ATCS-techathon-sona/hardware/esp32_controller.py) and the microcontroller sketch [eps32sketch.ino](file:///E:/Git%20Repo/ATCS-techathon-sona/hardware/eps32sketch.ino).
- **GUI (`gui/`)**: Displays feeds, charts, and overrides in [gui.py](file:///E:/Git%20Repo/ATCS-techathon-sona/gui/gui.py).

---

## 7. Tech Stack
- **Languages**: Python 3.10+, C++ (Arduino sketch)
- **Frameworks**: PyTorch 2.0+, Ultralytics YOLOv11m-seg, OpenCV 4.8
- **Hardware**: ESP32-WROOM-32, PySerial, LED Array
- **Libraries**: Tkinter (UI), Matplotlib (Charting), Multiprocessing (GIL bypass)

---

## 8. Results & Achievements
Evaluated via [benchmark.py](file:///E:/Git%20Repo/ATCS-techathon-sona/benchmark.py) against a 30s static cycle under rush-hour patterns:

| Metric | Static Cycle | ATCS (RL-Driven) | Difference |
| :--- | :--- | :--- | :--- |
| **Avg. Queue Volume** | 1,201 | 1,043 | **-13.2%** |
| **Emergency Delay** | 77.7s | 31.8s | **-59.1%** |
| **Wait Pressure** | 6,469 units | 4,189 units | **-35.2%** |
| **Throughput** | 900 units | 862 units | -4.3% (balanced) |

---

## 9. Future Enhancements
- **Corridor Coordination**: Multi-Agent RL to synchronize green lights across adjacent blocks.
- **Edge Deployment**: Optimize models with TensorRT for platforms like NVIDIA Jetson.
- **Weather Adaptation**: Train YOLO on rain, fog, and night conditions.
- **V2X Telemetry**: direct communication from vehicle transponders to override perception latency.

---

## 10. Conclusion
ATCS provides a scalable, queue-aware traffic control model. By combining YOLOv11 and D3QN decision-making in a multi-processed Python framework, it decreases traffic pressure by 35.2% and preempts emergency vehicles. Physical implementation on the ESP32 proves the system's readiness for next-generation smart cities.
