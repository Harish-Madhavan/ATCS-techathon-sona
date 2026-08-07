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

from core import config

from rl_model.environment import TrafficEnv, get_time_varying_arrival_rate
from rl_model.network import DuelingQNetwork


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
    replay_buffer_size=50000,
    learning_rate=1e-3,
    gamma=0.98,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.992,
    target_update_frequency=10,
    checkpoint_interval=500,
    device_str=None,
    seed=42,
    use_time_varying_arrival_rate=False
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    duration_options = duration_options if duration_options is not None else [10, 20, 30]
    vehicle_type_names = vehicle_type_names or ['car', 'truck']
    vehicle_weights = vehicle_weights or [1.0, 2.0]

    num_vehicle_types = len(vehicle_type_names)
    v_end = num_lanes * (num_vehicle_types if num_vehicle_types > 0 else 1)
    input_size = v_end + num_lanes + num_lanes + num_lanes
    output_size = num_lanes * len(duration_options)

    if getattr(torch, 'xpu', None) and torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Training on device: {device}")

    env = TrafficEnv(num_lanes=num_lanes,
                     vehicle_type_names=vehicle_type_names,
                     vehicle_weights=vehicle_weights,
                     duration_options=duration_options,
                     arrival_rate_per_lane=arrival_rate_per_lane,
                     max_queue=max_queue,
                     use_time_varying_arrival_rate=use_time_varying_arrival_rate,
                     total_steps=50)
    env.max_time = 900.0  # 15 mins of simulated time per training episode for 2x faster speed

    model = DuelingQNetwork(input_size, output_size).to(device)
    target_model = DuelingQNetwork(input_size, output_size).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_episodes, eta_min=1e-5)
    # Huber (Smooth L1) loss is more robust to large TD-error outliers than MSE;
    # it behaves like L2 near zero but falls back to L1 for large errors,
    # preventing explosive gradient updates during early training.
    criterion = nn.SmoothL1Loss()
    replay_buffer = collections.deque(maxlen=replay_buffer_size)
    epsilon = epsilon_start
    # Collect this many transitions before starting any gradient updates.
    # 5 × batch_size (min 2 000) ensures the buffer contains diverse states
    # before the network begins fitting, reducing early over-specialisation.
    replay_warmup = max(batch_size * 5, 2000)
    best_episode_reward = -float("inf")
    best_ckpt_path = os.path.join(os.path.dirname(config.RL_MODEL_PATH), "rl_model_best.pth")
    current_lr = learning_rate  # kept in sync after each scheduler step

    start_time = time.time()
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        # normalize state
        state = env.normalize_state(state, normalize_max)
        episode_reward = 0.0
        done = False
        episode_loss_sum = 0.0   # accumulates Huber loss for this episode
        episode_update_count = 0  # number of gradient steps taken this episode

        while not done:
            # Action masking logic based on environment's consecutive lane count
            num_duration_options = len(duration_options)
            invalid_lane = env.current_lane if env.consecutive_lane_count >= 2 else None

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            if random.random() < epsilon:
                if invalid_lane is not None:
                    valid_actions = [a for a in range(output_size) if (a // num_duration_options) != invalid_lane]
                    action = random.choice(valid_actions)
                else:
                    action = random.randint(0, output_size - 1)
            else:
                with torch.no_grad():
                    q_values = model(state_tensor)
                if invalid_lane is not None:
                    q_values_masked = q_values.clone()
                    for a in range(output_size):
                        if (a // num_duration_options) == invalid_lane:
                            q_values_masked[0, a] = -1e9
                    action = int(torch.argmax(q_values_masked).item())
                else:
                    action = int(torch.argmax(q_values).item())

            next_state, reward, done, info = env.step(action)
            next_state_normalized = env.normalize_state(next_state, normalize_max)
            episode_reward += reward

            # Track if there is any invalid lane in the next state for masking
            invalid_lane_next = env.current_lane if env.consecutive_lane_count >= 2 else -1
            replay_buffer.append((state, action, reward, next_state_normalized, done, invalid_lane_next))
            state = next_state_normalized

            if len(replay_buffer) >= replay_warmup:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones, next_invalid_lanes = zip(*batch)

                states_tensor = torch.FloatTensor(np.array(states)).to(device)
                next_states_tensor = torch.FloatTensor(np.array(next_states)).to(device)
                actions_tensor = torch.LongTensor(actions).to(device)
                rewards_tensor = torch.FloatTensor(rewards).to(device)
                dones_tensor = torch.FloatTensor(dones).to(device)
                next_invalid_lanes_tensor = torch.LongTensor(next_invalid_lanes).to(device)

                current_q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_vals = model(next_states_tensor).clone()
                    # Mask out invalid actions in the next state for Double DQN target calculation
                    for b_idx in range(batch_size):
                        invalid_l = next_invalid_lanes[b_idx]
                        if invalid_l != -1:
                            for a_idx in range(output_size):
                                if (a_idx // num_duration_options) == invalid_l:
                                    next_q_vals[b_idx, a_idx] = -1e9

                    next_actions = next_q_vals.argmax(1, keepdim=True)
                    next_q_values = target_model(next_states_tensor).gather(1, next_actions).squeeze(1)

                target_q_values = rewards_tensor + (gamma * next_q_values * (1 - dones_tensor))

                loss = criterion(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                # Tightened from 10 → 5: more conservative bound reduces the
                # risk of destructive large updates in early episodes.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                episode_loss_sum += loss.item()
                episode_update_count += 1

        # Step the LR scheduler only after warmup (mirrors the training condition)
        if len(replay_buffer) >= replay_warmup:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        if epsilon > epsilon_end:
            epsilon *= epsilon_decay

        # Save the best model observed so far (only post-warmup, so we don't
        # accidentally preserve a lucky random-policy episode).
        if episode_reward > best_episode_reward and len(replay_buffer) >= replay_warmup:
            best_episode_reward = episode_reward
            torch.save(model.state_dict(), best_ckpt_path)

        if episode % target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())

        if episode % 50 == 0 and epsilon < 0.5:
            # Only checkpoint once epsilon has decayed below 0.5 (policy is
            # meaningfully non-random). Filename encodes epsilon for traceability.
            ckpt_path = os.path.join(
                os.path.dirname(config.RL_MODEL_PATH),
                f"rl_model_ep{episode}_eps{epsilon:.3f}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path} (episode {episode}, eps={epsilon:.4f})", flush=True)

        if episode % 10 == 0:
            elapsed = time.time() - start_time
            avg_loss = episode_loss_sum / max(1, episode_update_count)
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Reward: {episode_reward:.2f} | Best: {best_episode_reward:.2f} | "
                f"eps: {epsilon:.4f} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | "
                f"Elapsed: {elapsed:.1f}s",
                flush=True
            )

    torch.save(model.state_dict(), config.RL_MODEL_PATH)
    print(f"Training complete. Model saved as {config.RL_MODEL_PATH}", flush=True)


if __name__ == '__main__':
    # Get parameters from config.py
    num_lanes = len(config.TRAFFIC_LIGHT_CONFIG['Intersection1']['phases'])
    vehicle_type_names = config.TARGET_CLASSES
    vehicle_weights = [config.VEHICLE_TYPE_WEIGHTS[vt] for vt in vehicle_type_names]
    duration_options = config.DURATION_OPTIONS

    print(f"[Train] Lanes: {num_lanes}, Vehicle types: {vehicle_type_names}")
    print(f"[Train] Duration options: {len(duration_options)} values, range [{min(duration_options)}-{max(duration_options)}]")
    print(f"[Train] Action space: {num_lanes * len(duration_options)} actions")

    train(
        num_lanes=num_lanes,
        vehicle_type_names=vehicle_type_names,
        vehicle_weights=vehicle_weights,
        duration_options=duration_options,
        num_episodes=800,
        epsilon_decay=0.992,
        gamma=0.98,
        use_time_varying_arrival_rate=True
    )