
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
        
        self.state = np.zeros((self.num_lanes, self.num_vehicle_types), dtype=np.int32)
        self.vehicle_weights = np.array([1, 2])  # Cars, Trucks

    def reset(self):
        self.state = np.random.randint(0, 10, size=(self.num_lanes, self.num_vehicle_types))
        return self.state.flatten()

    def step(self, action):
        lane = action // self.num_duration_options
        duration_index = action % self.num_duration_options
        duration = self.duration_options[duration_index]

        # Reward is the negative sum of waiting vehicles before the action
        reward = -np.sum(self.state * self.vehicle_weights)

        # --- Improved Vehicle Flow Logic ---
        # Define a vehicle flow rate (vehicles per second).
        vehicle_flow_rate = 0.5 
        
        # Calculate the total number of vehicles that can pass in the given duration.
        max_vehicles_to_pass = int(duration * vehicle_flow_rate)
        
        # Get the current number of vehicles in the chosen lane.
        vehicles_in_lane = self.state[lane]
        
        # Determine the actual number of vehicles that will pass, clearing them type by type.
        passed_total = 0
        vehicles_passed_per_type = np.zeros_like(vehicles_in_lane)

        for i in range(self.num_vehicle_types):
            can_pass = max_vehicles_to_pass - passed_total
            if can_pass <= 0:
                break
            
            num_waiting = vehicles_in_lane[i]
            num_passed = min(num_waiting, can_pass)
            
            vehicles_passed_per_type[i] = num_passed
            passed_total += num_passed
            
        self.state[lane] -= vehicles_passed_per_type
        # --- End of Improved Logic ---

        # New vehicles arrive randomly in all lanes
        self.state += np.random.randint(0, 3, size=(self.num_lanes, self.num_vehicle_types))

        # Add the negative sum of waiting vehicles *after* the action to the reward
        reward -= np.sum(self.state * self.vehicle_weights)

        done = False
        return self.state.flatten(), reward, done, {}

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
    input_size = num_lanes * num_vehicle_types
    output_size = num_lanes * len(duration_options)

    # Hyperparameters
    learning_rate = 0.001
    gamma = 0.9
    num_episodes = 20000
    batch_size = 256
    replay_buffer_size = 20000
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.99992 # Adjusted for longer training

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"Training on device: {device}")

    env = TrafficEnv(num_lanes=num_lanes, num_vehicle_types=num_vehicle_types, duration_options=duration_options)
    model = QNetwork(input_size, output_size).to(device)
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
                
                with torch.no_grad():
                    next_q_values = model(next_states_tensor).max(1)[0]
                
                target_q_values = rewards_tensor + (gamma * next_q_values * (1 - dones_tensor))

                loss = criterion(current_q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epsilon > epsilon_end:
            epsilon *= epsilon_decay

        reward_history.append(episode_reward)

        if episode % 100 == 0:
            last_100_rewards = reward_history[-100:]
            avg_reward = sum(last_100_rewards) / len(last_100_rewards)
            print(f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.4f}, Loss: {loss.item() if len(replay_buffer) > batch_size else 'N/A'}")

    torch.save(model.state_dict(), "rl_model.pth")
    print("Model trained and saved as rl_model.pth")

if __name__ == "__main__":
    train()
