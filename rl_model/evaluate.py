import torch

# Import from canonical module locations instead of the root train script
from rl_model.environment import TrafficEnv
from rl_model.network import load_q_network
from rl_model.device import get_best_device
from rl_model.contract import state_dim as _state_dim, action_dim as _action_dim


def evaluate():
    from core import config
    # Environment and model parameters (match root train defaults)
    num_lanes = len(config.TRAFFIC_LIGHT_CONFIG['Intersection1']['phases'])
    vehicle_type_names = config.TARGET_CLASSES
    num_vehicle_types = len(vehicle_type_names)
    duration_options = config.DURATION_OPTIONS
    
    input_size = _state_dim(num_lanes, num_vehicle_types)
    output_size = _action_dim(num_lanes, len(duration_options))
    num_episodes = 100

    device = get_best_device()
    print(f"Evaluating on device: {device}")

    # Pass all configured parameters to TrafficEnv
    vt_weights = [float(config.VEHICLE_TYPE_WEIGHTS.get(vt, 1.0)) for vt in vehicle_type_names]
    env = TrafficEnv(
        num_lanes=num_lanes,
        vehicle_type_names=vehicle_type_names,
        vehicle_weights=vt_weights,
        duration_options=duration_options
    )
    try:
        model = load_q_network(config.RL_MODEL_PATH, input_size, output_size, device)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"Could not load model: {e}")
        print("The model architecture has changed or file is missing. You may need to retrain.")
        return

    total_rewards = 0
    normalize_max = 50.0

    for episode in range(num_episodes):
        state = env.reset()
        # Normalize initial state to match training pipeline
        state = env.normalize_state(state, normalize_max=normalize_max)
        episode_reward = 0
        done = False
        step = 0
        # Use time-based termination (env.done) instead of a fixed step limit,
        # matching the training episode length (env.max_time = 1800 s by default).
        while not done:
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
            # Normalize next state before storing for next inference
            state = env.normalize_state(next_state, normalize_max=normalize_max)
            episode_reward += reward
            step += 1

        print(f"Episode {episode + 1}/{num_episodes} finished in {step} steps, reward: {episode_reward:.2f}")
        total_rewards += episode_reward

    average_reward = total_rewards / num_episodes
    print(f"\nAverage reward over {num_episodes} episodes: {average_reward:.4f}")


if __name__ == "__main__":
    evaluate()