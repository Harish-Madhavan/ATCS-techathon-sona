import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections
import time
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

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

class TrafficEnv:
    """
    Improved training environment with configurable vehicle types, per-lane arrival rates (Poisson),
    and per-type vehicle weights. State layout matches runtime: flattened per-lane per-type counts
    followed by a one-hot current-lane vector.
    """
    def __init__(self,
                 num_lanes=4,
                 vehicle_type_names=None,
                 vehicle_weights=None,
                 duration_options=None,
                 arrival_rate_per_lane=None,
                 max_queue=100,
                 use_time_varying_arrival_rate=False,
                 total_steps=50):
        self.num_lanes = num_lanes
        self.vehicle_type_names = vehicle_type_names or ['car', 'truck']
        self.num_vehicle_types = len(self.vehicle_type_names)
        self.vehicle_weights = np.array(vehicle_weights if vehicle_weights is not None else [1.0, 2.0])
        self.duration_options = duration_options or [10, 20, 30]
        self.num_duration_options = len(self.duration_options)
        self.action_space_size = self.num_lanes * self.num_duration_options
        self.use_time_varying_arrival_rate = use_time_varying_arrival_rate
        self.total_steps = total_steps
        self.current_step = 0

        # arrival_rate_per_lane: list of lambda (vehicles/sec) for Poisson arrivals; if a single float provided, broadcast
        if arrival_rate_per_lane is None:
            self.base_arrival_rate = 0.2
            arrival_rate_per_lane = [self.base_arrival_rate] * self.num_lanes
        else:
            self.base_arrival_rate = arrival_rate_per_lane if isinstance(arrival_rate_per_lane, (int, float)) else np.mean(arrival_rate_per_lane)

        if isinstance(arrival_rate_per_lane, (int, float)):
            arrival_rate_per_lane = [float(arrival_rate_per_lane)] * self.num_lanes
        self.arrival_rate_per_lane = list(arrival_rate_per_lane)

        self.max_queue = max_queue

        # internal state: integer counts per lane x vehicle type
        self.state_vehicle_counts = np.zeros((self.num_lanes, self.num_vehicle_types), dtype=np.int32)
        self.ambulance_presence = np.zeros(self.num_lanes, dtype=float)
        self.wait_times = np.zeros(self.num_lanes, dtype=float)
        self.current_lane = 0

    def get_one_hot_lane(self):
        one_hot = np.zeros(self.num_lanes, dtype=float)
        one_hot[self.current_lane] = 1.0
        return one_hot

    def normalize_state(self, state, normalize_max):
        # state is [vehicle_counts (24), wait_times (4), ambulance_flags (4), one_hot_lane (4)]
        v_end = self.num_lanes * self.num_vehicle_types
        w_end = v_end + self.num_lanes
        normalized = state.copy()
        normalized[:v_end] = normalized[:v_end] / max(1.0, float(normalize_max))
        # Normalize wait times by 100 seconds
        normalized[v_end:w_end] = normalized[v_end:w_end] / 100.0
        # ambulance flags and one-hot lane remain as 0/1
        return normalized

    def reset(self):
        # random initial queues
        self.state_vehicle_counts = np.random.randint(0, 10, size=(self.num_lanes, self.num_vehicle_types))
        self.ambulance_presence = np.zeros(self.num_lanes, dtype=float)
        self.wait_times = np.zeros(self.num_lanes, dtype=float)
        self.current_lane = 0
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        return np.concatenate([
            self.state_vehicle_counts.flatten().astype(float),
            self.wait_times,
            self.ambulance_presence,
            self.get_one_hot_lane()
        ])

    def step(self, action):
        self.current_step += 1
        if self.use_time_varying_arrival_rate:
            current_arrival_rate = get_time_varying_arrival_rate(self.current_step, self.total_steps, base_rate=self.base_arrival_rate)
            self.arrival_rate_per_lane = [current_arrival_rate] * self.num_lanes

        lane = int(action // self.num_duration_options)
        self.current_lane = lane
        duration_index = int(action % self.num_duration_options)
        duration = float(self.duration_options[duration_index])

        # Increment wait times for all lanes, then reset for the served lane
        for ln in range(self.num_lanes):
            if ln != lane:
                self.wait_times[ln] += duration
            else:
                self.wait_times[ln] = 0.0

        # If ambulance was in this lane, it's now served
        self.ambulance_presence[lane] = 0.0

        # Service capacity: vehicles-per-second * duration -> capacity units in 'weight' units
        # Here we interpret vehicle_weights as the relative 'space' consumed. We compute how many weight-units
        # can pass and allocate them greedy by vehicle type (heavier types first)
        vehicle_flow_rate = 0.5  # vehicles per second base; could be customized per-lane
        max_capacity_units = int(duration * vehicle_flow_rate)

        vehicles_in_lane = self.state_vehicle_counts[lane].copy()
        passed_total_units = 0
        vehicles_passed = np.zeros_like(vehicles_in_lane)

        # Serve heavier vehicles first (configurable choice)
        for i in np.argsort(self.vehicle_weights)[::-1]:
            if passed_total_units >= max_capacity_units:
                break
            available = int(vehicles_in_lane[i])
            capacity_left = max_capacity_units - passed_total_units
            # how many of this vehicle type can pass given their weight
            can_pass = capacity_left // int(max(1, round(self.vehicle_weights[i])))
            num_pass = min(available, can_pass)
            vehicles_passed[i] = num_pass
            passed_total_units += int(num_pass * self.vehicle_weights[i])

        self.state_vehicle_counts[lane] = np.maximum(0, self.state_vehicle_counts[lane] - vehicles_passed)

        # Arrivals: Poisson arrivals per lane per vehicle type proportional to arrival_rate and duration
        for ln in range(self.num_lanes):
            lam = max(0.0, float(self.arrival_rate_per_lane[ln]) * max(1.0, duration))
            # distribute arrivals across types uniformly with small randomness
            arrivals_total = np.random.poisson(lam)
            if arrivals_total > 0:
                # split arrivals proportionally to inverse weights (prefer lighter vehicles)
                weights_inv = 1.0 / (self.vehicle_weights + 1e-6)
                probs = weights_inv / np.sum(weights_inv)
                arrivals_by_type = np.random.multinomial(arrivals_total, probs)
            else:
                arrivals_by_type = np.zeros(self.num_vehicle_types, dtype=int)
            self.state_vehicle_counts[ln] += arrivals_by_type

            # Simulate ambulance arrivals
            if np.random.random() < 0.02: # 2% chance per lane per decision step
                self.ambulance_presence[ln] = 1.0

        # clamp queues
        self.state_vehicle_counts = np.minimum(self.state_vehicle_counts, self.max_queue)

        # Reward: penalize queues, wait times, and ESPECIALLY waiting ambulances
        queue_penalty = -float(np.sum(self.state_vehicle_counts * self.vehicle_weights))
        wait_penalty = -2.0 * float(np.sum(self.wait_times))
        amb_penalty = -1000.0 * np.sum(self.ambulance_presence) 
        reward = queue_penalty + wait_penalty + amb_penalty

        done = False
        next_state = self._get_state()
        return next_state, reward, done, {}


class DuelingQNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Value and Advantage heads
        self.value_head = nn.Linear(hidden_size, 1)
        self.adv_head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        v = self.value_head(x)
        a = self.adv_head(x)
        # combine: Q = V + (A - mean(A))
        return v + (a - a.mean(dim=1, keepdim=True))


def train(
    num_lanes=4,
    vehicle_type_names=None,
    vehicle_weights=None,
    duration_options=None,
    arrival_rate_per_lane=None,
    max_queue=100,
    normalize_max=50.0,
    num_episodes=5000,
    batch_size=128,
    replay_buffer_size=20000,
    learning_rate=1e-3,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.997,
    target_update_frequency=10,
    checkpoint_interval=500,
    device_str=None,
    seed=42,
    use_time_varying_arrival_rate=False
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    duration_options = duration_options
    vehicle_type_names = vehicle_type_names or ['car', 'truck']
    vehicle_weights = vehicle_weights or [1.0, 2.0]

    input_size = (num_lanes * len(vehicle_type_names)) + num_lanes + num_lanes + num_lanes
    output_size = num_lanes * len(duration_options)

    if(torch.xpu.is_available()):
        device= "xpu"
    elif(torch.cuda.is_available()):
        device = "cuda"
    else:
        device = "cpu"
    print(f"Training on device: {device}" )

    env = TrafficEnv(num_lanes=num_lanes,
                     vehicle_type_names=vehicle_type_names,
                     vehicle_weights=vehicle_weights,
                     duration_options=duration_options,
                     arrival_rate_per_lane=arrival_rate_per_lane,
                     max_queue=max_queue,
                     use_time_varying_arrival_rate=use_time_varying_arrival_rate,
                     total_steps=50)

    model = DuelingQNetwork(input_size, output_size).to(device)
    target_model = DuelingQNetwork(input_size, output_size).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    replay_buffer = collections.deque(maxlen=replay_buffer_size)
    epsilon = epsilon_start

    start_time = time.time()
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        # normalize state
        state = env.normalize_state(state, normalize_max)
        episode_reward = 0.0
        last_lane_chosen = None

        for step in range(50):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            if random.random() < epsilon:
                action = random.randint(0, output_size - 1)
            else:
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = int(torch.argmax(q_values).item())

            next_state, reward, done, _ = env.step(action)
            # apply a small penalty for choosing the same lane repeatedly to encourage fairness
            chosen_lane = int(action // env.num_duration_options)
            if last_lane_chosen is not None and chosen_lane == last_lane_chosen:
                repeat_penalty = -0.5  # tunable penalty per repeated selection
                reward += repeat_penalty
            last_lane_chosen = chosen_lane
            next_state = env.normalize_state(next_state, normalize_max)
            episode_reward += reward

            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.FloatTensor(np.array(states)).to(device)
                next_states_tensor = torch.FloatTensor(np.array(next_states)).to(device)
                actions_tensor = torch.LongTensor(actions).to(device)
                rewards_tensor = torch.FloatTensor(rewards).to(device)
                dones_tensor = torch.FloatTensor(dones).to(device)

                current_q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_actions = model(next_states_tensor).argmax(1, keepdim=True)
                    next_q_values = target_model(next_states_tensor).gather(1, next_actions).squeeze(1)

                target_q_values = rewards_tensor + (gamma * next_q_values * (1 - dones_tensor))

                loss = criterion(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epsilon > epsilon_end:
            epsilon *= epsilon_decay

        if episode % target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())

        if episode % checkpoint_interval == 0:
            torch.save(model.state_dict(), f"rl_model_ep{episode}.pth")
            print(f"Checkpoint saved: rl_model_ep{episode}.pth (episode {episode})")

        if episode % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode}/{num_episodes}, Episode Reward: {episode_reward:.2f}, Epsilon: {epsilon:.4f}, Elapsed: {elapsed:.1f}s")

    torch.save(model.state_dict(), "rl_model.pth")
    print("Training complete. Model saved as rl_model.pth")


if __name__ == '__main__':
    # Get parameters from config.py
    num_lanes = len(config.TRAFFIC_LIGHT_CONFIG['Intersection1']['phases'])
    vehicle_type_names = config.TARGET_CLASSES
    vehicle_weights = [config.VEHICLE_TYPE_WEIGHTS[vt] for vt in vehicle_type_names]
    duration_options = config.DURATION_OPTIONS

    train(
        num_lanes=num_lanes,
        vehicle_type_names=vehicle_type_names,
        vehicle_weights=vehicle_weights,
        duration_options=duration_options,
        num_episodes=2000,
        use_time_varying_arrival_rate=True
    )