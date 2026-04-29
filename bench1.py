import torch
import numpy as np
import argparse
import statistics
import config
import collections

# Using the TrafficEnv as defined in your train.py
from train import TrafficEnv
from train import DuelingQNetwork


def get_time_varying_arrival_rate(step, total_steps, base_rate=0.1, peak_multiplier=4.0):
    """
    Simulates a rush hour pattern using a sinusoidal wave.
    Starts at base_rate, peaks in the middle, and returns to base_rate.
    """
    # Scale step to be in the range [0, 2*pi]
    radian_step = (step / total_steps) * 2 * np.pi
    # Use a sine wave shifted and scaled to create a single peak
    multiplier = 1 + (peak_multiplier - 1) * (-np.cos(radian_step) + 1) / 2
    return base_rate * multiplier


def run_simulation(env, agent, num_steps=1000, scenario='normal', base_arrival_rate=0.2):
    """
    Runs a single, more realistic simulation and collects detailed metrics.
    """
    state = env.reset()
    
    # Metrics to track
    total_reward = 0
    max_queue_len = 0
    all_queue_lengths = []

    # Scenario setup
    incident_triggered = False
    incident_step = num_steps // 2
    
    for step in range(num_steps):
        # Apply dynamic traffic patterns
        current_arrival_rate = get_time_varying_arrival_rate(step, num_steps, base_rate=base_arrival_rate)
        env.arrival_rate_per_lane = [current_arrival_rate] * env.num_lanes

        # Apply incident scenario
        if scenario == 'incident' and not incident_triggered and step >= incident_step:
            incident_lane = np.random.randint(0, env.num_lanes)
            surge = np.random.randint(15, 25, size=env.num_vehicle_types)
            print(f"\nInjecting incident surge of {np.sum(surge)} vehicles into lane {incident_lane} at step {step}...")
            env.state_vehicle_counts[incident_lane] += surge
            incident_triggered = True

        # Agent takes action and environment responds
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        
        total_reward += reward
        state = next_state

        # Update metrics
        current_queues = env.state_vehicle_counts
        max_queue_len = max(max_queue_len, np.max(current_queues))
        all_queue_lengths.append(np.sum(current_queues))

    final_vehicles = np.sum(env.state_vehicle_counts)
    
    return {
        "avg_wait_time": -total_reward / num_steps, # Average number of vehicles waiting per step
        "max_queue_len": max_queue_len,
        "p95_wait_time": np.percentile(all_queue_lengths, 95) if all_queue_lengths else 0,
        "final_queue_len": final_vehicles
    }


class RLAgent:
    """Agent that uses a trained Q-network to make decisions."""
    def __init__(self, model_path, input_size, output_size, device, normalize_max=50.0):
        ckpt = torch.load(model_path, map_location=device)
        self.model = DuelingQNetwork(input_size, output_size)
        self.model.load_state_dict(ckpt)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.normalize_max = max(1.0, float(normalize_max))

    def choose_action(self, state):
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
        try:
            self.duration_index = self.duration_options.index(fixed_duration)
        except ValueError:
            self.duration_index = min(range(len(self.duration_options)), key=lambda i: abs(self.duration_options[i] - fixed_duration))
        self.current_lane = 0

    def choose_action(self, state):
        action = self.current_lane * self.num_duration_options + self.duration_index
        self.current_lane = (self.current_lane + 1) % self.num_lanes
        return action

def analyze_results(results):
    """Helper function to compute statistics from a list of result dictionaries."""
    if not results:
        return {key: (0, 0) for key in ["avg_wait_time", "max_queue_len", "p95_wait_time", "final_queue_len"]}
    
    metrics = collections.defaultdict(list)
    for res in results:
        for key, value in res.items():
            # CHANGE THIS LINE: Convert NumPy numbers to Python floats
            metrics[key].append(float(value))
    
    stats = {}
    for key, values in metrics.items():
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        stats[key] = (mean, std)
    return stats


def print_results(agent_name, stats):
    """Helper function to print formatted statistics."""
    print(f"--- {agent_name} ---")
    print(f"  Avg Wait Time (vehicles/step): {stats['avg_wait_time'][0]:.2f} (±{stats['avg_wait_time'][1]:.2f})")
    print(f"  Max Queue Length (vehicles):   {stats['max_queue_len'][0]:.2f} (±{stats['max_queue_len'][1]:.2f})")
    print(f"  P95 Wait Time (vehicles):      {stats['p95_wait_time'][0]:.2f} (±{stats['p95_wait_time'][1]:.2f})")
    print(f"  Final Queue Length (vehicles): {stats['final_queue_len'][0]:.2f} (±{stats['final_queue_len'][1]:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Benchmark RL model against a static agent under realistic conditions.")
    parser.add_argument("--model_path", type=str, default="rl_model.pth", help="Path to the trained RL model file.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps per simulation run.")
    parser.add_argument("--num_runs", type=int, default=20, help="Number of simulation runs for statistical analysis.")
    parser.add_argument("--num_lanes", type=int, default=4, help="Number of lanes.")
    parser.add_argument("--normalize_max", type=float, default=50.0, help="Normalization max used in training.")
    parser.add_argument("--scenario", type=str, default="rush_hour", choices=["rush_hour", "incident"], help="The simulation scenario to run.")
    parser.add_argument("--base_arrival_rate", type=float, default=0.15, help="Base vehicle arrival rate for dynamic patterns.")
    
    args = parser.parse_args()

    # --- Benchmark Configuration (MUST MATCH TRAINING) ---
    # The error message tells us the exact configuration of the saved model.
    # We will hardcode it here to ensure the benchmark script can load it.
    
    # Input mismatch (12 vs 28) implies 2 vehicle types were used during training.
    # The default from the original train.py was ['car', 'truck'].
    vehicle_type_names = ['car', 'truck'] 
    
    # We must also get the correct weights for only these types.
    # Let's pull them from the config, but only for the types we need.
    vehicle_weights = [config.VEHICLE_TYPE_WEIGHTS.get(v, 1.0) for v in vehicle_type_names]

    # Output mismatch (12 vs 224) implies 3 duration options were used during training.
    # The default from the original train.py was [10, 20, 30].
    duration_options = [10, 20, 30]
    
    num_lanes = args.num_lanes

    # Now, calculate the sizes based on the *correct* training configuration.
    input_size = (num_lanes * len(vehicle_type_names)) + num_lanes
    output_size = num_lanes * len(duration_options)

    print("--- Benchmark Configuration ---")
    print(f"Using configuration matching the trained model:")
    print(f"Vehicle Types: {vehicle_type_names} (Input Size: {input_size})")
    print(f"Duration Options: {duration_options} (Output Size: {output_size})")
    print("-----------------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialize Agents ---
    # This will now succeed because input_size and output_size are correct.
    rl_agent = RLAgent(args.model_path, input_size, output_size, device, normalize_max=args.normalize_max)
    static_agent = StaticAgent(num_lanes, duration_options, fixed_duration=30)
    
    agents = {
        "RL Agent": rl_agent,
        "Static Agent": static_agent,
    }
    
    all_results = collections.defaultdict(list)

    print(f"\nRunning {args.num_runs} simulations of {args.num_steps} steps each.")
    print(f"Scenario: {args.scenario.replace('_', ' ').title()}")
    print("-" * 40)

    for i in range(args.num_runs):
        # Randomize base arrival rate for each run to simulate day-to-day variance
        run_base_rate = np.random.uniform(args.base_arrival_rate * 0.8, args.base_arrival_rate * 1.2)
        print(f"Run {i+1}/{args.num_runs} (Base Arrival Rate: {run_base_rate:.3f})...")
        
        for agent_name, agent in agents.items():
            # Create a fresh environment for each agent run to ensure fair comparison
            # The environment MUST also use the same configuration.
            env = TrafficEnv(
                num_lanes=num_lanes,
                vehicle_type_names=vehicle_type_names,
                vehicle_weights=vehicle_weights,
                duration_options=duration_options
            )
            
            result = run_simulation(env, agent, args.num_steps, args.scenario, run_base_rate)
            all_results[agent_name].append(result)

    # --- Calculate Statistics and Display Results ---
    print("\n--- Benchmark Results ---")
    
    rl_stats = analyze_results(all_results["RL Agent"])
    static_stats = analyze_results(all_results["Static Agent"])
    
    print_results("RL Agent", rl_stats)
    print_results("Static Agent", static_stats)
    
    print("-" * 40)
    
    # --- Comparative Analysis ---
    rl_perf = rl_stats['avg_wait_time'][0]
    static_perf = static_stats['avg_wait_time'][0]
    
    if static_perf > 0:
        improvement = ((static_perf - rl_perf) / static_perf) * 100
        print(f"Improvement of RL over Static Agent (based on Avg Wait Time): {improvement:.2f}%")
    else:
        print("Static agent had zero waiting time, cannot calculate improvement.")


if __name__ == "__main__":
    main()