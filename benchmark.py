import torch
import numpy as np
import argparse
import statistics
import os
import json
from core import config

from rl_model.environment import TrafficEnv
from rl_model.device import get_best_device
from rl_model.agents import RLAgent, StaticAgent, GreedyAgent

def run_simulation(env, agent):
    """
    Run one episode of simulation until done (simulated time exhausted).
    Returns a dictionary of metrics.
    """
    # Reset any stateful agent (e.g. StaticAgent lane counter) so every episode
    # starts from the same configuration for reproducible, fair comparisons.
    if hasattr(agent, 'reset'):
        agent.reset()
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
        next_state, reward, done, info = env.step(action)
        
        # Read throughput from info dict with fallback
        if isinstance(info, dict) and "cleared_value" in info:
            cleared_value = info["cleared_value"]
        else:
            # Fallback calculation
            current_queue_weight = float(np.sum(env.state_vehicle_counts * env.vehicle_weights))
            cleared_value = max(0.0, reward + current_queue_weight * 0.025)
            
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

def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark RL model against a static timer agent and greedy preemption agent.")
    parser.add_argument("--model_path", type=str, default=config.RL_MODEL_PATH, help="Path to the trained RL model file.")
    parser.add_argument("--num_runs", type=int, default=30, help="Number of simulation runs.")
    parser.add_argument("--num_lanes", type=int, default=4, help="Number of lanes.")
    parser.add_argument("--normalize_max", type=float, default=50.0, help="Normalization max.")
    parser.add_argument("--use_model_action_space", action='store_true', help="Auto-adjust action space.")
    parser.add_argument("--min_duration", type=int, default=5, help="Min duration.")
    parser.add_argument("--max_duration", type=int, default=60, help="Max duration.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to export benchmark results as JSON.")
    args = parser.parse_args(argv)

    num_lanes = int(args.num_lanes)
    try:
        vehicle_type_names = getattr(config, 'TARGET_CLASSES', list(config.VEHICLE_TYPE_WEIGHTS.keys()))
    except:
        vehicle_type_names = ['car', 'truck']

    duration_options = getattr(config, 'DURATION_OPTIONS', [10, 20, 30])
    num_vehicle_types = len(vehicle_type_names)
    v_end = num_lanes * num_vehicle_types
    input_size = v_end + num_lanes + num_lanes + num_lanes
    output_size = num_lanes * len(duration_options)

    device = get_best_device()
    
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
            print(f"Adapted duration options from model: {duration_options}")
    except:
        pass

    try:
        vt_weights = [float(config.VEHICLE_TYPE_WEIGHTS.get(vt, 1.0)) for vt in vehicle_type_names]
    except:
        vt_weights = [1.0, 2.0]

    env = TrafficEnv(num_lanes=num_lanes, vehicle_type_names=vehicle_type_names, vehicle_weights=vt_weights, duration_options=duration_options, use_time_varying_arrival_rate=True)
    
    rl_agent = None
    model_loaded = False
    try:
        rl_agent = RLAgent(args.model_path, input_size, output_size, device, normalize_max=args.normalize_max)
        model_loaded = True
    except Exception as e:
        print(f"\nWarning: Could not load RL model at '{args.model_path}' ({e}).\nBenchmarking will compare baselines (Static vs Greedy Preemption) only.")

    static_agent = StaticAgent(num_lanes, duration_options, fixed_duration=30)
    greedy_agent = GreedyAgent(num_lanes, duration_options, fixed_duration=20)

    print(f"\nRunning {args.num_runs} simulation episodes...")
    
    # Evaluate Static Agent
    np.random.seed(42)
    torch.manual_seed(42)
    env.seed(42)
    static_results = [run_simulation(env, static_agent) for _ in range(args.num_runs)]

    # Evaluate Greedy Agent
    np.random.seed(42)
    torch.manual_seed(42)
    env.seed(42)
    greedy_results = [run_simulation(env, greedy_agent) for _ in range(args.num_runs)]

    # Evaluate RL Agent
    if model_loaded:
        np.random.seed(42)
        torch.manual_seed(42)
        env.seed(42)
        rl_results = [run_simulation(env, rl_agent) for _ in range(args.num_runs)]
    else:
        rl_results = None

    def get_stats(results):
        keys = results[0].keys()
        return {k: (statistics.mean([r[k] for r in results]), statistics.stdev([r[k] for r in results])) for k in keys}

    static_stats = get_stats(static_results)
    greedy_stats = get_stats(greedy_results)
    rl_stats = get_stats(rl_results) if model_loaded else None

    print("\n" + "=" * 90)
    print(" " * 25 + "ADAPTIVE TRAFFIC CONTROL SYSTEM BENCHMARK")
    print("=" * 90)
    print(f"{'Metric':<25} | {'Static (30s)':<18} | {'Greedy (Preempt)':<18} | {'ATCS (RL)':<18}")
    print("-" * 90)
    
    metrics = [
        ("avg_queue_len", "Avg Queue Length"),
        ("max_wait_time", "Max Wait Time (s)"),
        ("throughput", "Throughput (Weight)"),
        ("emergency_delay", "Emergency Delay (s)")
    ]
    for key, name in metrics:
        s_val, s_std = static_stats[key]
        g_val, g_std = greedy_stats[key]
        
        s_str = f"{s_val:<7.2f} +/- {s_std:<5.1f}"
        g_str = f"{g_val:<7.2f} +/- {g_std:<5.1f}"
        
        if model_loaded:
            r_val, r_std = rl_stats[key]
            r_str = f"{r_val:<7.2f} +/- {r_std:<5.1f}"
        else:
            r_str = "N/A"
            
        print(f"{name:<25} | {s_str:<18} | {g_str:<18} | {r_str:<18}")
    
    s_reward, s_r_std = static_stats['total_reward']
    g_reward, g_r_std = greedy_stats['total_reward']
    s_reward_str = f"{s_reward:<7.1f} +/- {s_r_std:<5.1f}"
    g_reward_str = f"{g_reward:<7.1f} +/- {g_r_std:<5.1f}"
    
    if model_loaded:
        r_reward, r_r_std = rl_stats['total_reward']
        r_reward_str = f"{r_reward:<7.1f} +/- {r_r_std:<5.1f}"
    else:
        r_reward_str = "N/A"
        
    print("-" * 90)
    print(f"{'Average Episode Reward':<25} | {s_reward_str:<18} | {g_reward_str:<18} | {r_reward_str:<18}")
    print("=" * 90)
    
    # Print improvement percentages if RL was loaded
    if model_loaded:
        print("\nImprovement of ATCS (RL) over baselines:")
        for key, name in metrics:
            s_val = static_stats[key][0]
            g_val = greedy_stats[key][0]
            r_val = rl_stats[key][0]
            
            # Lower is better for all except throughput
            is_throughput = (key == "throughput")
            
            # Over static
            if s_val != 0:
                diff_s = ((r_val - s_val) / s_val * 100)
                if not is_throughput:
                    diff_s = -diff_s  # positive diff means improvement
                s_imp_str = f"{diff_s:+.1f}%"
            else:
                s_imp_str = "N/A"
                
            # Over greedy
            if g_val != 0:
                diff_g = ((r_val - g_val) / g_val * 100)
                if not is_throughput:
                    diff_g = -diff_g  # positive diff means improvement
                g_imp_str = f"{diff_g:+.1f}%"
            else:
                g_imp_str = "N/A"
                
            print(f"  * {name:<23}: {s_imp_str:<10} better than Static  |  {g_imp_str:<10} better than Greedy")

    if args.output_json:
        export_data = {
            "num_runs": args.num_runs,
            "static": {k: {"mean": float(v[0]), "std": float(v[1])} for k, v in static_stats.items()},
            "greedy": {k: {"mean": float(v[0]), "std": float(v[1])} for k, v in greedy_stats.items()},
        }
        if rl_stats:
            export_data["rl"] = {k: {"mean": float(v[0]), "std": float(v[1])} for k, v in rl_stats.items()}
        try:
            out_dir = os.path.dirname(args.output_json)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(export_data, f, indent=2)
            print(f"\n[Benchmark] Results successfully exported to: {args.output_json}")
        except Exception as e_json:
            print(f"\n[Benchmark Error] Failed to export JSON results: {e_json}")

if __name__ == "__main__":
    main()