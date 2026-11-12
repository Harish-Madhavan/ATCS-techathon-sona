# Adaptive Traffic Control System (ATCS)

An autonomous, queue-aware traffic signal control system combining **Computer Vision (YOLOv11)**, **Deep Reinforcement Learning (Dueling Double Deep Q-Network - D3QN)**, and **Microcontroller Hardware-in-the-loop (ESP32)**.

---

## 🚦 System Architecture

```
                                 [ Camera Feeds ]
                                        │
                                        ▼
               ┌──────────────────────────────────────────────────┐
               │    Perception Engine (vision/video_processor.py) │
               │   • Point-in-Polygon ROI Lane Checks            │
               │   • YOLOv11 Vehicle Detection & Classification   │
               │   • Ambulance & Siren Emergency Detection        │
               └────────────────────────┬─────────────────────────┘
                                        │ (Multiprocessing Queue)
                                        ▼
               ┌──────────────────────────────────────────────────┐
               │    Cognition & D3QN Agent (rl_model/)            │
               │   • 36-dim Normalized State Vector               │
               │   • Dueling Q-Network (LayerNorm + Ortho Init)   │
               │   • Action Masking & Consecutive Repeat Limiter  │
               └────────────────────────┬─────────────────────────┘
                                        │
                                        ▼
               ┌──────────────────────────────────────────────────┐
               │    Traffic FSM Controller (core/)                │
               │   • Safety Transitions (GREEN → YELLOW → RED)    │
               │   • Realtime Flow & Gap-Out Extensions           │
               │   • Instant Emergency Preemption Override        │
               └────────────┬─────────────────────────┬───────────┘
                            │                         │
                            ▼                         ▼
         ┌───────────────────────────────┐ ┌───────────────────────────────┐
         │ Operator Dashboard (gui/)     │ │ Microcontroller Bridge (hardware/)│
         │ • Dark Theme UI & Overrides   │ │ • ESP32 Serial Protocol      │
         │ • Realtime Matplotlib Trends  │ │ • Physical 4-Way Traffic LEDs │
         └───────────────────────────────┘ └───────────────────────────────┘
```

---

## 📁 Repository Structure

```
ATCS-techathon-sona/
├── core/                       # Finite State Machine and controller logic
│   ├── config.py               # Global timings, classes, and system configuration
│   ├── intersection_state.py   # State machine, timers, and preemption transitions
│   ├── traffic_logic.py        # Controller interface & RL state vector generation
│   └── log.py                  # Structured application logging
├── vision/                     # Computer vision & multiprocessing workers
│   ├── polygon_utils.py        # Interactive ROI polygon definer
│   └── video_processor.py      # Multiprocessed YOLO inference pipeline
├── rl_model/                   # Reinforcement Learning decision engine
│   ├── network.py              # Dueling Q-Network architecture (+ checkpoint I/O)
│   ├── environment.py          # Queue dynamics, Poisson arrivals & reward simulation
│   ├── inference.py            # Realtime RL inference, masking & starvation guard
│   ├── train.py                # D3QN training algorithm & CLI
│   ├── agents.py               # Benchmark agents (RL / static / greedy)
│   ├── contract.py             # Shared state/action contract & mask helpers
│   ├── device.py               # Accelerator selection (xpu > cuda > cpu)
│   └── evaluate.py             # Evaluation routines
├── hardware/                   # Physical microcontroller interface
│   ├── esp32_controller.py     # Serial driver with MOCK support
│   └── firmware/               # Microcontroller firmware
│       ├── esp32sketch.ino     # Arduino C++ sketch for ESP32 LED actuation
│       └── README.md           # Firmware build & flashing guide
├── gui/                        # Operator interface & dashboard
│   ├── gui.py                  # Main application window
│   ├── gui_components.py       # Traffic light canvas & live trend plots
│   ├── gui_setup.py            # Welcome configuration dialog
│   ├── gui_worker_manager.py   # Background process manager
│   ├── gui_rl_handler.py       # Deprecated shim → rl_model.inference
│   └── theme.py                # Shared dark-theme design system
├── models/                     # Neural network checkpoints
│   ├── rl_model.pth            # Trained D3QN weights
│   ├── rl_model_best.pth       # Best reward snapshot
│   ├── yoloe-11m-seg.pt        # General vehicle segmentation model
│   ├── last.pt                 # Ambulance detection model
│   └── checkpoints/            # Intermediate training snapshots (ep50..800)
├── tests/                      # Automated regression test suite
│   ├── test_traffic_logic.py   # FSM, preemption & RL state unit tests
│   ├── test_rl_environment.py  # Simulation dynamics & network unit tests
│   ├── test_rl_contract.py     # State/action contract & mask tests
│   ├── test_train.py           # RL training pipeline tests
│   ├── test_hardware.py        # Serial communication & mock unit tests
│   ├── test_q.py               # Model evaluation test
│   └── smoke_test.py           # End-to-end integration smoke test
├── tools/                      # CLI tools & offline inspection
│   ├── rl_model_inspect.py     # Q-value bias & state inspection tool
│   └── debug/                  # Diagnostic and repro scripts
├── docs/                       # Technical reports and documentation
│   ├── ATCS_Codebase_Report.md # Full system architecture report
│   └── Project_Report.md       # Comprehensive project documentation
├── benchmark.py                # Comparative benchmark suite (Static vs Greedy vs RL)
├── train.py                    # D3QN training CLI entrypoint (wraps rl_model.train)
├── main.py                     # Primary application entrypoint
├── pyproject.toml              # Standard Python project packaging & test config
└── requirements.txt            # Python dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

```bash
# Clone the repository
git clone https://github.com/Harish-Madhavan/ATCS-techathon-sona.git
cd ATCS-techathon-sona

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the System

```bash
# Launch interactive GUI Operator Dashboard
python main.py

# Launch in Headless Mode (no UI window)
python main.py --headless

# Launch with ESP32 on a specific port (or MOCK mode)
python main.py --esp32 COM3
python main.py --esp32 MOCK

# Run using a specific test video across all approaches
python main.py --video assets/vid1.mp4
```

---

## 🧪 Automated Testing

Run the full automated test suite using Python `unittest`:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run the GUI smoke test:
```bash
python tests/smoke_test.py
```

---

## 📊 Benchmarking & Training

### Benchmark RL vs Baselines
Compare the trained RL agent against fixed 30s static cycles and greedy preemption agents:

```bash
# Run 30 simulation episodes and export results to JSON
python benchmark.py --num_runs 30 --output_json benchmark_results.json
```

### Train the D3QN Model
Train the reinforcement learning agent in the simulation environment:

```bash
python train.py
```

---

## 🔌 Hardware Setup (ESP32)

1. Flash `hardware/firmware/esp32sketch.ino` onto an ESP32 board using Arduino IDE.
2. Connect 4-way traffic light LEDs to the mapped GPIO pins (configured in `hardware/firmware/esp32sketch.ino`).
3. Set `ESP32_ENABLED = True` and `ESP32_PORT = "COM3"` (or pass `--esp32 COM3` in `main.py`).
