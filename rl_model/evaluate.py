import torch
import torch.nn as nn
import numpy as np

class TrafficEnv:
    def __init__(self, num_lanes=4, num_vehicle_types=2):
        self.num_lanes = num_lanes
        self.num_vehicle_types = num_vehicle_types
        
        self.state = np.zeros((self.num_lanes, self.num_vehicle_types), dtype=np.int32)
        self.vehicle_weights = np.array([1, 2])  # Cars, Trucks
        self.cars_per_second = 0.5

    def reset(self):
        self.state = np.random.randint(0, 10, size=(self.num_lanes, self.num_vehicle_types))
        return self.state.flatten()

    def step(self, action):
        lane, duration = action
        duration = max(0, duration)

        reward = -np.sum(self.state * self.vehicle_weights)

        passed_vehicles_per_type = np.floor(duration * self.cars_per_second / self.num_vehicle_types).astype(int)
        if passed_vehicles_per_type > 0:
            passed_vehicles = np.random.randint(0, passed_vehicles_per_type + 1, size=(self.num_vehicle_types))
        else:
            passed_vehicles = np.zeros(self.num_vehicle_types, dtype=int)
        self.state[lane] = np.maximum(0, self.state[lane] - passed_vehicles)

        self.state += np.random.randint(0, 3, size=(self.num_lanes, self.num_vehicle_types))

        reward -= np.sum(self.state * self.vehicle_weights)

        done = False
        return self.state.flatten(), reward, done, {}

class QNetwork(nn.Module):
    def __init__(self, input_size, num_lanes):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_lanes * 2)
        self.num_lanes = num_lanes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.num_lanes, 2)

def evaluate():
    # Environment and model parameters
    num_lanes = 4
    num_vehicle_types = 2
    input_size = num_lanes * num_vehicle_types
    num_episodes = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    env = TrafficEnv(num_lanes=num_lanes, num_vehicle_types=num_vehicle_types)
    model = QNetwork(input_size, num_lanes).to(device)
    try:
        model.load_state_dict(torch.load("rl_model.pth"))
    except RuntimeError as e:
        print(f"Could not load model state dict: {e}")
        print("The model architecture has changed. You need to retrain the model.")
        return
        
    model.eval()

    total_rewards = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(50):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                model_output = model(state_tensor)
            
            q_values = model_output[:, :, 0]
            durations = model_output[:, :, 1]
            
            lane = torch.argmax(q_values).item()
            duration = durations[0, lane].item()
            
            duration = max(5.0, min(duration, 60.0))

            print(f"  Step {step}: Green light for Lane {lane} for {duration:.2f} seconds")

            action = (lane, duration)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
        
        total_rewards += episode_reward

    average_reward = total_rewards / num_episodes
    print(f"\nAverage reward over {num_episodes} episodes: {average_reward}")

if __name__ == "__main__":
    evaluate()