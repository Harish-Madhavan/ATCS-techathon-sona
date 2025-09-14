import torch
import numpy as np
from train import TrafficEnv, QNetwork
import argparse
import statistics

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
    def __init__(self, model_path, input_size, output_size, device):
        self.model = QNetwork(input_size, output_size)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.device = device

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

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
    args = parser.parse_args()

    # --- Environment and Model Parameters (should match training) ---
    num_lanes = 4
    num_vehicle_types = 2
    duration_options = [10, 20, 30]
    input_size = (num_lanes * num_vehicle_types) + num_lanes
    output_size = num_lanes * len(duration_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialize Environment and Agents ---
    env = TrafficEnv(num_lanes=num_lanes, num_vehicle_types=num_vehicle_types, duration_options=duration_options)
    rl_agent = RLAgent(args.model_path, input_size, output_size, device)
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