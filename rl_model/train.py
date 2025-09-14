import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections

class TrafficEnv:
    def __init__(self, num_lanes=4, num_vehicle_types=2, duration_options=[10, 20, 30]):
        self.num_lanes = num_lanes
        self.num_vehicle_types = num_vehicle_types
        self.duration_options = duration_options
        self.num_duration_options = len(duration_options)
        self.action_space_size = self.num_lanes * self.num_duration_options
        
        self.state_vehicle_counts = np.zeros((self.num_lanes, self.num_vehicle_types), dtype=np.int32)
        self.vehicle_weights = np.array([1, 2])  # Cars, Trucks
        self.current_lane = 0 # Start with lane 0 being notionally active

    def get_one_hot_lane(self):
        one_hot = np.zeros(self.num_lanes)
        one_hot[self.current_lane] = 1
        return one_hot

    def reset(self):
        self.state_vehicle_counts = np.random.randint(0, 10, size=(self.num_lanes, self.num_vehicle_types))
        self.current_lane = 0
        return np.concatenate([self.state_vehicle_counts.flatten(), self.get_one_hot_lane()])

    def step(self, action):
        lane = action // self.num_duration_options
        self.current_lane = lane # Update the active lane
        duration_index = action % self.num_duration_options
        duration = self.duration_options[duration_index]

        # --- Vehicle Flow Logic ---
        vehicle_flow_rate = 0.5  # vehicles per second
        max_vehicles_to_pass = int(duration * vehicle_flow_rate)
        
        vehicles_in_lane = self.state_vehicle_counts[lane].copy()
        
        passed_total = 0
        vehicles_passed_per_type = np.zeros_like(vehicles_in_lane)

        for i in np.argsort(self.vehicle_weights)[::-1]:
            can_pass = max_vehicles_to_pass - passed_total
            if can_pass <= 0:
                break
            
            num_waiting = vehicles_in_lane[i]
            num_passed = min(num_waiting, can_pass // self.vehicle_weights[i])
            
            vehicles_passed_per_type[i] = num_passed
            passed_total += num_passed * self.vehicle_weights[i]
            
        self.state_vehicle_counts[lane] -= vehicles_passed_per_type.astype(np.int32)
        
        self.state_vehicle_counts += np.random.randint(0, 3, size=(self.num_lanes, self.num_vehicle_types))
        
        self.state_vehicle_counts = np.maximum(0, self.state_vehicle_counts)

        reward = -np.sum(self.state_vehicle_counts)

        done = False
        next_state = np.concatenate([self.state_vehicle_counts.flatten(), self.get_one_hot_lane()])
        return next_state, reward, done, {}

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train():
    # Environment and model parameters
    num_lanes = 4
    num_vehicle_types = 2
    duration_options = [10, 20, 30]
    # State size is now vehicle counts + one-hot encoded lane
    input_size = (num_lanes * num_vehicle_types) + num_lanes 
    output_size = num_lanes * len(duration_options)

    # Hyperparameters
    learning_rate = 0.001
    gamma = 0.9
    num_episodes = 20000  # Increased episodes
    batch_size = 256
    replay_buffer_size = 20000
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.99995 # Adjusted for longer training
    target_update_frequency = 10 # For DDQN

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"Training on device: {device}")

    env = TrafficEnv(num_lanes=num_lanes, num_vehicle_types=num_vehicle_types, duration_options=duration_options)
    
    model = QNetwork(input_size, output_size).to(device)
    target_model = QNetwork(input_size, output_size).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    replay_buffer = collections.deque(maxlen=replay_buffer_size)
    epsilon = epsilon_start
    
    reward_history = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(50):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            if random.random() < epsilon:
                action = random.randint(0, output_size - 1)
            else:
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state

            if len(replay_buffer) > batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.FloatTensor(np.array(states)).to(device)
                next_states_tensor = torch.FloatTensor(np.array(next_states)).to(device)
                actions_tensor = torch.LongTensor(actions).to(device)
                rewards_tensor = torch.FloatTensor(rewards).to(device)
                dones_tensor = torch.FloatTensor(dones).to(device)

                current_q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                
                # --- DDQN Update ---
                with torch.no_grad():
                    # Select actions with the online model
                    next_actions = model(next_states_tensor).argmax(1, keepdim=True)
                    # Evaluate actions with the target model
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

        reward_history.append(episode_reward)

        if episode % 100 == 0:
            last_100_rewards = reward_history[-100:]
            avg_reward = sum(last_100_rewards) / len(last_100_rewards)
            print(f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.4f}, Loss: {loss.item() if len(replay_buffer) > batch_size else 'N/A'}")

    torch.save(model.state_dict(), "rl_model.pth")
    print("Model trained and saved as rl_model.pth")

if __name__ == "__main__":
    train()