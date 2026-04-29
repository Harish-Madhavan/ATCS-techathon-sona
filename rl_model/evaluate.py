import torch
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import TrafficEnv, DuelingQNetwork


def evaluate():
    # Environment and model parameters (match root train defaults)
    num_lanes = 4
    num_vehicle_types = 6 # Standardized to match config.TARGET_CLASSES
    duration_options = list(range(5, 61))
    input_size = (num_lanes * num_vehicle_types) + num_lanes + num_lanes + num_lanes
    num_episodes = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    env = TrafficEnv(num_lanes=num_lanes, duration_options=duration_options)
    model = DuelingQNetwork(input_size, num_lanes * len(duration_options)).to(device)
    try:
        model.load_state_dict(torch.load("rl_model.pth", map_location=device))
    except (RuntimeError, FileNotFoundError) as e:
        print(f"Could not load model: {e}")
        print("The model architecture has changed or file is missing. You may need to retrain.")
        return

    model.eval()

    total_rewards = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(50):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)

            # q_values is shaped (1, output_size)
            action = int(torch.argmax(q_values).item())
            lane = action // len(duration_options)
            duration_index = action % len(duration_options)
            duration = duration_options[duration_index]

            print(f"  Step {step}: Green light for Lane {lane} for {duration} seconds")

            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward

        total_rewards += episode_reward

    average_reward = total_rewards / num_episodes
    print(f"\nAverage reward over {num_episodes} episodes: {average_reward}")


if __name__ == "__main__":
    evaluate()