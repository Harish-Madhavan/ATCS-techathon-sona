import torch
import numpy as np
from benchmark import TrafficEnv, RLAgent
from config import TARGET_CLASSES, VEHICLE_TYPE_WEIGHTS, DURATION_OPTIONS

num_lanes = 4
vehicle_type_names = TARGET_CLASSES
vt_weights = [float(VEHICLE_TYPE_WEIGHTS.get(vt, 1.0)) for vt in vehicle_type_names]
duration_options = DURATION_OPTIONS
v_end = num_lanes * len(vehicle_type_names)
input_size = v_end + 3 * num_lanes
output_size = num_lanes * len(duration_options)

device = torch.device("cpu")
env = TrafficEnv(num_lanes=num_lanes, vehicle_type_names=vehicle_type_names, vehicle_weights=vt_weights, duration_options=duration_options)
agent = RLAgent("rl_model.pth", input_size, output_size, device)

state = env.reset()
normalized_state = env.normalize_state(state, agent.normalize_max)
state_tensor = torch.FloatTensor(np.array(normalized_state, dtype=float)).unsqueeze(0).to(device)
with torch.no_grad():
    q_values = agent.model(state_tensor)

print("Q Values Shape:", q_values.shape)
print("Q Values:", q_values)
print("Chosen Action:", int(torch.argmax(q_values).item()))
