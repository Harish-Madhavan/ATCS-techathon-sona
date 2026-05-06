import torch
import numpy as np
import argparse
import statistics
import config

from train import TrafficEnv, DuelingQNetwork

def run_simulation(env, agent):
    """
    Run one episode of simulation until done (simulated time exhausted).
    Returns a dictionary of metrics.
    """
    state = env.reset()
    state = env.normalize_state(state, agent.normalize_max)
    total_reward = 0
    done = False
    
    # Metrics tracking
    queue_lengths = []
    max_wait_times = []
    emergency_delays = []
    
    # Track when ambulances arrive
    ambulance_arrival_times = {} # lane_idx -> [arrival_time, ...]
    
    total_cleared = 0
    
    while not done:
        # Before step, record arrivals
        for ln in range(env.num_lanes):
            if env.ambulance_presence[ln] > 0:
                if ln not in ambulance_arrival_times:
                    ambulance_arrival_times[ln] = env.current_time

        action = agent.choose_action(state, env)
        
        # Step the environment
        # We need to know how many vehicles passed. 
        # Since env.step doesn't return it, let's look at how it calculates it.
        # vehicles_passed = min(available, can_pass) ...
        # Let's modify env.step to return this, or just calculate it here.
        # Actually, let's just use the reward cleared_value as a proxy if we can.
        # Or better, just sum up the weight of vehicles cleared.
        
        # To get the actual count, I'd have to modify train.py.
        # Let's check if I can modify train.py safely.
        
        next_state, reward, done, _ = env.step(action)
        
        # We can estimate vehicles cleared from reward if we know the penalty.
        # Reward = cleared_value - current_queue_weight * 0.1
        # cleared_value = reward + current_queue_weight * 0.1
        current_queue_weight = float(np.sum(env.state_vehicle_counts * env.vehicle_weights))
        cleared_value = reward + current_queue_weight * 0.1
        total_cleared += cleared_value

        next_state = env.normalize_state(next_state, agent.normalize_max)
        
        post_counts = env.state_vehicle_counts.copy()
        queue_lengths.append(np.sum(post_counts))
        max_wait_times.append(np.max(env.wait_times))
        
        total_reward += reward
        state = next_state
        
        for ln in range(env.num_lanes):
            if ln in ambulance_arrival_times and env.ambulance_presence[ln] == 0:
                delay = env.current_time - ambulance_arrival_times[ln]
                emergency_delays.append(delay)
                del ambulance_arrival_times[ln]

    return {
        "total_reward": total_reward,
        "avg_queue_len": np.mean(queue_lengths),
        "max_wait_time": np.max(max_wait_times),
        "emergency_delay": np.mean(emergency_delays) if emergency_delays else 0,
        "throughput": total_cleared
    }

class RLAgent:
    """Agent that uses a trained Q-network to make decisions."""
    def __init__(self, model_path, input_size, output_size, device, normalize_max=50.0):
        
        ckpt = torch.load(model_path, map_location=device)
        
        inferred_output = None
        if isinstance(ckpt, dict):
            for k in ('adv_head.weight', 'adv_head.bias', 'value_head.weight', 'value_head.bias'):
                if k in ckpt:
                    param = ckpt[k]
                    if hasattr(param, 'shape'):
                        if len(param.shape) >= 1:
                            inferred_output = int(param.shape[0])
                            break
        if inferred_output is not None and inferred_output != output_size:
            raise RuntimeError(f"Model mismatch. Use --use_model_action_space.")

        self.model = DuelingQNetwork(input_size, output_size)
        self.model.load_state_dict(ckpt)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.normalize_max = max(1.0, float(normalize_max))

    def choose_action(self, state, env=None):
        state_tensor = torch.FloatTensor(np.array(state, dtype=float)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values).item())

class StaticAgent:
    """Agent that cycles through lanes with a fixed green light duration."""
    def __init__(self, num_lanes, duration_options, fixed_duration=30, normalize_max=50.0):
        self.num_lanes = num_lanes
        self.duration_options = duration_options
        self.num_duration_options = len(duration_options)
        self.fixed_duration = fixed_duration
        self.normalize_max = max(1.0, float(normalize_max))
        try:
            self.duration_index = self.duration_options.index(self.fixed_duration)
        except ValueError:
            self.duration_index = min(range(len(self.duration_options)), key=lambda i: abs(self.duration_options[i] - self.fixed_duration))
        self.current_lane = 0

    def choose_action(self, state, env=None):
        action = self.current_lane * self.num_duration_options + self.duration_index
        self.current_lane = (self.current_lane + 1) % self.num_lanes
        return action

def main():
    parser = argparse.ArgumentParser(description="Benchmark RL model against a static timer agent.")
    parser.add_argument("--model_path", type=str, default="rl_model.pth", help="Path to the trained RL model file.")
    parser.add_argument("--num_runs", type=int, default=30, help="Number of simulation runs.")
    parser.add_argument("--num_lanes", type=int, default=4, help="Number of lanes.")
    parser.add_argument("--normalize_max", type=float, default=50.0, help="Normalization max.")
    parser.add_argument("--use_model_action_space", action='store_true', help="Auto-adjust action space.")
    parser.add_argument("--min_duration", type=int, default=5, help="Min duration.")
    parser.add_argument("--max_duration", type=int, default=60, help="Max duration.")
    args = parser.parse_args()

    num_lanes = int(args.num_lanes)
    try:
        vehicle_type_names = list(config.VEHICLE_TYPE_WEIGHTS.keys())
    except:
        vehicle_type_names = ['car', 'truck']

    duration_options = getattr(config, 'DURATION_OPTIONS', [10, 20, 30])
    num_vehicle_types = len(vehicle_type_names)
    v_end = num_lanes * num_vehicle_types
    input_size = v_end + num_lanes + num_lanes + num_lanes
    output_size = num_lanes * len(duration_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        ckpt = torch.load(args.model_path, map_location=device)
        inferred_output = None
        if isinstance(ckpt, dict):
            for k in ('adv_head.weight', 'adv_head.bias'):
                if k in ckpt:
                    inferred_output = int(ckpt[k].shape[0])
                    break
        if inferred_output is not None and inferred_output != output_size:
            num_duration_options = inferred_output // num_lanes
            min_d, max_d = int(args.min_duration), int(args.max_duration)
            step = (max_d - min_d) / (num_duration_options - 1) if num_duration_options > 1 else 0
            duration_options = [int(round(min_d + i * step)) for i in range(num_duration_options)]
            output_size = inferred_output
            print(f"Adapted duration options: {duration_options}")
    except:
        pass

    try:
        vt_weights = [float(config.VEHICLE_TYPE_WEIGHTS.get(vt, 1.0)) for vt in vehicle_type_names]
    except:
        vt_weights = [1.0, 2.0]

    env = TrafficEnv(num_lanes=num_lanes, vehicle_type_names=vehicle_type_names, vehicle_weights=vt_weights, duration_options=duration_options, use_time_varying_arrival_rate=True)
    rl_agent = RLAgent(args.model_path, input_size, output_size, device, normalize_max=args.normalize_max)
    static_agent = StaticAgent(num_lanes, duration_options, fixed_duration=30)

    print(f"\nRunning {args.num_runs} episodes...")
    rl_results = [run_simulation(env, rl_agent) for _ in range(args.num_runs)]
    static_results = [run_simulation(env, static_agent) for _ in range(args.num_runs)]

    def get_stats(results):
        keys = results[0].keys()
        return {k: (statistics.mean([r[k] for r in results]), statistics.stdev([r[k] for r in results])) for k in keys}

    rl_stats = get_stats(rl_results)
    static_stats = get_stats(static_results)

    print("\n--- Benchmark Results ---")
    print("-" * 75)
    print(f"{'Metric':<25} | {'Static (30s)':<15} | {'ATCS (RL)':<15} | {'Diff (%)':<10}")
    print("-" * 75)
    
    metrics = [
        ("avg_queue_len", "Avg Queue Length"),
        ("max_wait_time", "Max Wait Time (s)"),
        ("throughput", "Throughput (Weight)"),
        ("emergency_delay", "Emergency Delay (s)")
    ]
    for key, name in metrics:
        s_val, r_val = static_stats[key][0], rl_stats[key][0]
        diff = ((r_val - s_val) / s_val * 100) if s_val != 0 else 0
        print(f"{name:<25} | {s_val:<15.2f} | {r_val:<15.2f} | {diff:<10.2f}%")
    
    s_press, r_press = -static_stats['total_reward'][0], -rl_stats['total_reward'][0]
    press_diff = ((r_press - s_press) / s_press * 100) if s_press != 0 else 0
    print(f"{'Wait Pressure (Reward)':<25} | {s_press:<15.2f} | {r_press:<15.2f} | {press_diff:<10.2f}%")
    print("-" * 75)

if __name__ == "__main__":
    main()