import torch
import numpy as np
import argparse
import statistics
import config

from train import TrafficEnv, DuelingQNetwork

def run_simulation(env, agent, num_steps=1000):
    """Runs a single simulation for a given number of steps."""
    state = env.reset()
    total_reward = 0
    for _ in range(num_steps):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    # The reward is negative waiting time, so we return the negative total reward.
    return -total_reward

class RLAgent:
    """Agent that uses a trained Q-network to make decisions."""
    def __init__(self, model_path, input_size, output_size, device, normalize_max=50.0):
        # load checkpoint first to inspect output size
        ckpt = torch.load(model_path, map_location=device)
        # If checkpoint is a state_dict directly, find an output-sized parameter to infer output dim
        inferred_output = None
        if isinstance(ckpt, dict):
            # state_dict keys example: 'adv_head.weight' or 'value_head.weight'
            for k in ('adv_head.weight', 'adv_head.bias', 'value_head.weight', 'value_head.bias'):
                if k in ckpt:
                    param = ckpt[k]
                    if hasattr(param, 'shape'):
                        # adv_head.weight shape [out, hidden]
                        if len(param.shape) >= 1:
                            inferred_output = int(param.shape[0])
                            break
        if inferred_output is not None and inferred_output != output_size:
            raise RuntimeError(f"Model at '{model_path}' has output size {inferred_output} but benchmark expected {output_size}. Use --use_model_action_space to auto-adjust the environment action space to the model.")

        self.model = DuelingQNetwork(input_size, output_size)
        self.model.load_state_dict(ckpt)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.normalize_max = max(1.0, float(normalize_max))

    def choose_action(self, state):
        # Normalize state the same way as during training
        state_arr = np.array(state, dtype=float) / self.normalize_max
        state_tensor = torch.FloatTensor(state_arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values).item())

class StaticAgent:
    """Agent that cycles through lanes with a fixed green light duration."""
    def __init__(self, num_lanes, duration_options, fixed_duration=30):
        self.num_lanes = num_lanes
        self.duration_options = duration_options
        self.num_duration_options = len(duration_options)
        self.fixed_duration = fixed_duration
        try:
            # Find the index of the desired fixed duration.
            self.duration_index = self.duration_options.index(self.fixed_duration)
        except ValueError:
            # If the exact duration is not available, choose the closest one.
            self.duration_index = min(range(len(self.duration_options)), key=lambda i: abs(self.duration_options[i] - self.fixed_duration))
        self.current_lane = 0

    def choose_action(self, state):
        action = self.current_lane * self.num_duration_options + self.duration_index
        self.current_lane = (self.current_lane + 1) % self.num_lanes
        return action

def main():
    parser = argparse.ArgumentParser(description="Benchmark RL model against a static timer agent.")
    parser.add_argument("--model_path", type=str, default="rl_model.pth", help="Path to the trained RL model file.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps per simulation run.")
    parser.add_argument("--num_runs", type=int, default=30, help="Number of simulation runs for statistical analysis.")
    parser.add_argument("--num_lanes", type=int, default=4, help="Number of lanes/intersections (used to compute action space).")
    parser.add_argument("--vehicle_types", type=str, default="", help="Comma-separated vehicle type names (e.g. car,truck). If empty, derived from config keys or defaults to car,truck.")
    parser.add_argument("--normalize_max", type=float, default=50.0, help="Normalization max used in training.")
    parser.add_argument("--use_model_action_space", action='store_true', help="If set, derive environment action-space (duration options) from the model checkpoint output size.")
    parser.add_argument("--min_duration", type=int, default=5, help="Minimum duration (seconds) when auto-generating duration options from model output size.")
    parser.add_argument("--max_duration", type=int, default=60, help="Maximum duration (seconds) when auto-generating duration options from model output size.")
    args = parser.parse_args()

    # --- Environment and Model Parameters (should match training) ---
    num_lanes = int(args.num_lanes)
    # Determine vehicle types: CLI -> config -> fallback
    if args.vehicle_types:
        vehicle_type_names = [v.strip() for v in args.vehicle_types.split(',') if v.strip()]
    else:
        try:
            vehicle_type_names = list(config.VEHICLE_TYPE_WEIGHTS.keys())[:2]
            if not vehicle_type_names:
                vehicle_type_names = ['car', 'truck']
        except Exception:
            vehicle_type_names = ['car', 'truck']

    duration_options = getattr(config, 'DURATION_OPTIONS', [10, 20, 30])
    input_size = (num_lanes * len(vehicle_type_names)) + num_lanes
    output_size = num_lanes * len(duration_options)

    # Decide device early so we can inspect checkpoints with correct map_location
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Try to inspect the model checkpoint and auto-adapt the action space when needed.
    try:
        ckpt = torch.load(args.model_path, map_location=device)
        inferred_output = None
        if isinstance(ckpt, dict):
            for k in ('adv_head.weight', 'adv_head.bias', 'value_head.weight', 'value_head.bias'):
                if k in ckpt:
                    param = ckpt[k]
                    if hasattr(param, 'shape'):
                        inferred_output = int(param.shape[0])
                        break
        if inferred_output is not None and inferred_output != output_size:
            # If the user explicitly requested use_model_action_space or we detect a mismatch, adapt.
            if args.use_model_action_space or True:
                if inferred_output % num_lanes != 0:
                    raise RuntimeError(f"Inferred model output size {inferred_output} is not divisible by num_lanes={num_lanes}; cannot derive duration options.")
                num_duration_options = inferred_output // num_lanes
                min_d = int(args.min_duration)
                max_d = int(args.max_duration)
                if num_duration_options == 1:
                    duration_options = [min_d]
                else:
                    step = (max_d - min_d) / (num_duration_options - 1)
                    duration_options = [int(round(min_d + i * step)) for i in range(num_duration_options)]
                output_size = inferred_output
                print(f"Adapted duration options from model checkpoint: {duration_options} (num options: {num_duration_options})")
    except FileNotFoundError:
        print(f"Model file {args.model_path} not found; proceeding with config duration options.")
    except Exception as e:
        raise RuntimeError(f"Failed to inspect/adapt action space from model: {e}")

    # --- Initialize Environment and Agents ---
    # derive vehicle weights from config if available
    try:
        vt_weights = [float(config.VEHICLE_TYPE_WEIGHTS.get(vt, 1.0)) for vt in vehicle_type_names]
    except Exception:
        vt_weights = None

    env = TrafficEnv(num_lanes=num_lanes, vehicle_type_names=vehicle_type_names, vehicle_weights=vt_weights, duration_options=duration_options)
    rl_agent = RLAgent(args.model_path, input_size, output_size, device, normalize_max=args.normalize_max)
    static_agent = StaticAgent(num_lanes, duration_options, fixed_duration=30)

    # --- Run Benchmark Simulations ---
    print(f"\nRunning {args.num_runs} simulations of {args.num_steps} steps each...")
    
    rl_waiting_times = []
    for i in range(args.num_runs):
        print(f"  RL Agent Run {i+1}/{args.num_runs}...", end='\r')
        waiting_time = run_simulation(env, rl_agent, args.num_steps)
        rl_waiting_times.append(waiting_time)
    print("\nRL agent simulations complete.")

    static_waiting_times = []
    for i in range(args.num_runs):
        print(f"  Static Agent Run {i+1}/{args.num_runs}...", end='\r')
        waiting_time = run_simulation(env, static_agent, args.num_steps)
        static_waiting_times.append(waiting_time)
    print("\nStatic agent simulations complete.")

    # --- Calculate Statistics ---
    # Convert numpy types to float for compatibility with statistics module
    rl_waiting_times_float = [float(x) for x in rl_waiting_times]
    static_waiting_times_float = [float(x) for x in static_waiting_times]

    rl_mean = statistics.mean(rl_waiting_times_float)
    rl_std = statistics.stdev(rl_waiting_times_float) if args.num_runs > 1 else 0

    static_mean = statistics.mean(static_waiting_times_float)
    static_std = statistics.stdev(static_waiting_times_float) if args.num_runs > 1 else 0

    # --- Display Results ---
    print("\n--- Benchmark Results ---")
    print(f"Number of simulation runs: {args.num_runs}")
    print(f"Steps per simulation: {args.num_steps}")
    print("-" * 30)
    print("RL Agent:")
    print(f"  Mean Total Waiting Time: {rl_mean:.2f}")
    print(f"  Standard Deviation:      {rl_std:.2f}")
    print("-" * 30)
    print("Static Agent (30s):")
    print(f"  Mean Total Waiting Time: {static_mean:.2f}")
    print(f"  Standard Deviation:      {static_std:.2f}")
    print("-" * 30)

    if static_mean > 0:
        improvement = ((static_mean - rl_mean) / static_mean) * 100
        print(f"Improvement of RL over Static: {improvement:.2f}%")
    else:
        print("Static agent had zero waiting time, cannot calculate improvement.")


if __name__ == "__main__":
    main()