"""Dueling Double Deep Q-Network (D3QN) training pipeline for ATCS.

Supports:
- Action masking with consecutive repeat limiter
- Experience replay with warmup
- Cosine annealing learning rate scheduler
- Huber (Smooth L1) loss with gradient clipping
- Automatic hardware selection (Intel XPU, CUDA, or CPU)
- Safe checkpointing in `models/checkpoints/`
"""
import collections
import os
import random
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core import config
from rl_model.contract import state_dim as _state_dim, action_dim as _action_dim
from rl_model.device import get_best_device_str
from rl_model.environment import TrafficEnv
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
    use_time_varying_arrival_rate=False,
    checkpoint_dir=None,
    model_output_path=None,
):
    """Train D3QN agent in the simulated traffic intersection environment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    default_duration_options = getattr(config, 'DURATION_OPTIONS', [10, 20, 30])
    default_target_classes = getattr(config, 'TARGET_CLASSES', ['car', 'truck'])
    default_weights_dict = getattr(config, 'VEHICLE_TYPE_WEIGHTS', {'car': 1.0, 'truck': 2.0})
    default_weights_list = [float(default_weights_dict.get(vt, 1.0)) for vt in default_target_classes]

    duration_options = duration_options if duration_options is not None else default_duration_options
    vehicle_type_names = vehicle_type_names or default_target_classes
    vehicle_weights = vehicle_weights or default_weights_list

    output_model_path = model_output_path or config.RL_MODEL_PATH
    models_root = os.path.dirname(output_model_path)
    ckpts_dir = checkpoint_dir or os.path.join(models_root, "checkpoints")
    os.makedirs(ckpts_dir, exist_ok=True)

    input_size = _state_dim(num_lanes, len(vehicle_type_names))
    output_size = _action_dim(num_lanes, len(duration_options))

    device = device_str or get_best_device_str()
    print(f"Training on device: {device}")

    env = TrafficEnv(
        num_lanes=num_lanes,
        vehicle_type_names=vehicle_type_names,
        vehicle_weights=vehicle_weights,
        duration_options=duration_options,
        arrival_rate_per_lane=arrival_rate_per_lane,
        max_queue=max_queue,
        use_time_varying_arrival_rate=use_time_varying_arrival_rate,
        total_steps=50,
    )
    env.seed(seed)
    env.max_time = 900.0  # 15 mins of simulated time per training episode

    model = DuelingQNetwork(input_size, output_size).to(device)
    target_model = DuelingQNetwork(input_size, output_size).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_episodes, eta_min=1e-5)
    criterion = nn.SmoothL1Loss()
    replay_buffer = collections.deque(maxlen=replay_buffer_size)
    epsilon = epsilon_start

    replay_warmup = max(batch_size * 5, 2000)
    best_episode_reward = -float("inf")
    best_ckpt_path = os.path.join(models_root, "rl_model_best.pth")
    current_lr = learning_rate

    start_time = time.time()
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        state = env.normalize_state(state, normalize_max)
        episode_reward = 0.0
        done = False
        episode_loss_sum = 0.0
        episode_update_count = 0

        while not done:
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

                current_q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_vals = model(next_states_tensor).clone()
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                episode_loss_sum += loss.item()
                episode_update_count += 1

        if len(replay_buffer) >= replay_warmup:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        if epsilon > epsilon_end:
            epsilon *= epsilon_decay

        if episode_reward > best_episode_reward and len(replay_buffer) >= replay_warmup:
            best_episode_reward = episode_reward
            torch.save(model.state_dict(), best_ckpt_path)

        if episode % target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())

        if episode % 50 == 0 and epsilon < 0.5:
            ckpt_path = os.path.join(
                ckpts_dir,
                f"rl_model_ep{episode}_eps{epsilon:.3f}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path} (episode {episode}, eps={epsilon:.4f})", flush=True)

        if episode % 10 == 0 or episode == 1:
            elapsed = time.time() - start_time
            avg_loss = episode_loss_sum / max(1, episode_update_count)
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Reward: {episode_reward:.2f} | Best: {best_episode_reward:.2f} | "
                f"eps: {epsilon:.4f} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | "
                f"Elapsed: {elapsed:.1f}s",
                flush=True
            )

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    print(f"Training complete. Model saved as {output_model_path}", flush=True)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train ATCS D3QN Agent")
    parser.add_argument("--episodes", type=int, default=800, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--decay", type=float, default=0.992, help="Epsilon decay rate")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--device", type=str, default=None, help="Device to run training on (cpu, cuda, xpu)")
    parser.add_argument("--static-rates", action="store_true", help="Disable time-varying arrival rates")
    parser.add_argument("--output", type=str, default=None, help="Path to save final model weights")
    args = parser.parse_args(argv)

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
        num_episodes=args.episodes,
        epsilon_decay=args.decay,
        gamma=args.gamma,
        device_str=args.device,
        seed=args.seed,
        use_time_varying_arrival_rate=not args.static_rates,
        model_output_path=args.output,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
