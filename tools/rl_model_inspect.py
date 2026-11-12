"""Simple RL model inspector to reproduce Q-value biases outside the GUI.

Usage (PowerShell):
python .\tools\rl_model_inspect.py

It will load the model at `config.RL_MODEL_PATH` and run forward passes on:
- zero state
- one-hot lane states
- small random states

It prints q-values, argmax, and top-k entries.
"""
import os
import sys
import pathlib
import numpy as np
import torch

# Ensure the repository root is on sys.path so we can import project modules
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import config
from rl_model.network import DuelingQNetwork
from rl_model.device import get_best_device

DEVICE = get_best_device()


def load_checkpoint(path):
    if not os.path.exists(path):
        print(f"Model file not found: {path}")
        return None
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    # Unwrap new metadata format {version, input_size, output_size, state_dict}.
    if isinstance(ckpt, dict) and "state_dict" in ckpt and "fc1.weight" not in ckpt:
        return ckpt["state_dict"]
    return ckpt


def infer_model_dims_from_ckpt(ckpt, num_lanes):
    inferred_input = None
    inferred_output = None
    if isinstance(ckpt, dict):
        # infer output size from adv_head.bias/weight or value_head
        for k in ('adv_head.bias', 'adv_head.weight', 'value_head.bias', 'value_head.weight'):
            if k in ckpt:
                p = ckpt[k]
                if hasattr(p, 'shape'):
                    inferred_output = int(p.shape[0])
                    break
        # infer input from fc1.weight
        for k in ('fc1.weight', 'module.fc1.weight', 'model.fc1.weight'):
            if k in ckpt:
                p = ckpt[k]
                if hasattr(p, 'shape') and len(p.shape) >= 2:
                    inferred_input = int(p.shape[1])
                    break
        # fallback scan
        if inferred_input is None:
            cand = None
            for k, p in ckpt.items():
                try:
                    if hasattr(p, 'shape') and len(p.shape) == 2:
                        s0, s1 = int(p.shape[0]), int(p.shape[1])
                        if cand is None or s1 < cand[0]:
                            cand = (s1, k)
                except Exception:
                    continue
            if cand is not None:
                inferred_input = int(cand[0])
    return inferred_input, inferred_output


if __name__ == '__main__':
    ckpt = load_checkpoint(config.RL_MODEL_PATH)
    num_lanes = len(config.TRAFFIC_LIGHT_CONFIG['Intersection1']['phases'])
    inferred_input, inferred_output = infer_model_dims_from_ckpt(ckpt, num_lanes)
    print(f"Inferred input: {inferred_input}, inferred output: {inferred_output}")

    # choose runtime input size as current traffic controller would produce
    # approximate runtime vehicle types by inspecting config VEHICLE_TYPE_WEIGHTS
    num_vehicle_types = len(getattr(config, 'VEHICLE_TYPE_WEIGHTS', {}).keys()) if getattr(config, 'VEHICLE_TYPE_WEIGHTS', None) else 6
    runtime_input = num_lanes * num_vehicle_types + num_lanes + num_lanes + num_lanes
    print(f"Runtime input size (approx): {runtime_input} (lanes={num_lanes}, vehicle_types={num_vehicle_types}, wait_times={num_lanes}, ambulances={num_lanes})")

    # choose output size based on config duration options unless inferred_output exists
    duration_options = getattr(config, 'DURATION_OPTIONS', [10,20,30])
    default_output = len(duration_options) * num_lanes
    out_size = inferred_output if inferred_output is not None else default_output
    print(f"Using output_size={out_size}")

    # build model matching inferred input if possible, else runtime
    model_input = inferred_input if inferred_input is not None else runtime_input
    print(f"Building model with input_size={model_input}, output_size={out_size}")

    model = DuelingQNetwork(model_input, out_size).to(DEVICE)
    if ckpt is not None:
        # attempt full load, then safe partial
        try:
            model.load_state_dict(ckpt)
            print("Full checkpoint loaded successfully.")
        except Exception as e:
            print(f"Full load failed: {e}. Trying partial load by shape.")
            norm = {}
            for k, v in ckpt.items():
                nk = k
                if k.startswith('module.'):
                    nk = k[len('module.'):]
                norm[nk] = v
            model_state = model.state_dict()
            filtered = {}
            for k, v in norm.items():
                if k in model_state and hasattr(v, 'shape') and tuple(v.shape) == tuple(model_state[k].shape):
                    filtered[k] = v
            if filtered:
                model.load_state_dict(filtered, strict=False)
                print(f"Partial load ok, loaded {len(filtered)} params.")
            else:
                print("No matching params to load; proceeding with random init.")
    model.eval()

    def run_state(name, arr):
        t = torch.FloatTensor(np.array(arr, dtype=float)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q = model(t).cpu().numpy()[0]
        idx = int(np.argmax(q))
        topk = np.argsort(q)[-8:][::-1]
        print(f"State={name} -> argmax={idx} topk={[ (int(i), float(q[i])) for i in topk ]}")

    # zero state
    zero = np.zeros(model_input, dtype=float)
    run_state('zero', zero)

    # one-hot per-lane at tail
    one_hot = np.zeros(model_input, dtype=float)
    # set one-hot for lane 0
    if model_input >= num_lanes:
        one_hot[-num_lanes] = 1.0
    run_state('one_hot', one_hot)

    # small random states
    for i in range(3):
        r = np.random.randn(model_input) * 0.5
        run_state(f'rand{i}', r)

    # Now emulate GUI remapping from runtime input (e.g., 36) to model input (e.g., 12)
    def remap_runtime_to_model(runtime_state, model_input_size, num_lanes):
        runtime_len = len(runtime_state)
        # Assuming runtime state is [lanes*vtypes, lanes_wait, lanes_amb, lanes_onehot]
        runtime_num_vehicle_type = (runtime_len - 3 * num_lanes) // num_lanes
        
        # Check if model input looks like it has wait times and ambulance flags
        # Structure: lanes*vtypes + lanes (wait) + lanes (amb) + lanes (onehot)
        has_wait_and_amb = (model_input_size % num_lanes == 0) and ((model_input_size // num_lanes) >= 4)
        
        if has_wait_and_amb:
             model_num_vehicle_type = (model_input_size // num_lanes) - 3
        else:
             # fallback to old structure or intermediate
             model_num_vehicle_type = (model_input_size // num_lanes) - 1

        per_type_flat = np.array(runtime_state[:num_lanes * runtime_num_vehicle_type], dtype=float)
        wait_times = np.array(runtime_state[num_lanes * runtime_num_vehicle_type : num_lanes * (runtime_num_vehicle_type + 1)], dtype=float)
        amb_flags = np.array(runtime_state[num_lanes * (runtime_num_vehicle_type + 1) : num_lanes * (runtime_num_vehicle_type + 2)], dtype=float)
        one_hot = np.array(runtime_state[num_lanes * (runtime_num_vehicle_type + 2) : ], dtype=float)
        
        per_type = per_type_flat.reshape((num_lanes, runtime_num_vehicle_type))

        if runtime_num_vehicle_type == model_num_vehicle_type:
            new_v_flat = per_type.flatten()
        elif runtime_num_vehicle_type > model_num_vehicle_type:
            ratio = max(1, runtime_num_vehicle_type // max(1, model_num_vehicle_type))
            new = np.zeros((num_lanes, model_num_vehicle_type), dtype=float)
            for i in range(model_num_vehicle_type):
                start = i * ratio
                end = start + ratio
                new[:, i] = per_type[:, start:end].sum(axis=1)
            new_v_flat = new.flatten()
        else:
            new = np.zeros((num_lanes, model_num_vehicle_type), dtype=float)
            for i in range(model_num_vehicle_type):
                src = int((i * runtime_num_vehicle_type) / model_num_vehicle_type)
                new[:, i] = per_type[:, min(src, runtime_num_vehicle_type - 1)]
            new_v_flat = new.flatten()

        if has_wait_and_amb:
            mapped = np.concatenate([new_v_flat, wait_times, amb_flags, one_hot])
        else:
            mapped = np.concatenate([new_v_flat, one_hot])
            
        if len(mapped) < model_input_size:
            pad = np.zeros(model_input_size - len(mapped), dtype=float)
            mapped = np.concatenate([mapped, pad])
        else:
            mapped = mapped[:model_input_size]
        return mapped

    print('\n--- Remapping tests (simulate runtime states) ---')
    runtime_input_size = runtime_input
    tests = {}
    tests['runtime_zero'] = np.zeros(runtime_input_size, dtype=float)
    
    # heavy traffic on lane 0, high wait on lane 2
    lane0 = np.zeros(runtime_input_size, dtype=float)
    vt = num_vehicle_types
    for t in range(vt):
        lane0[t] = 10.0  # lane 0 counts
    lane0[num_lanes * vt + 2] = 120.0 # high wait time on lane 2
    # also add ambulance on lane 1
    lane0[num_lanes * vt + num_lanes + 1] = 1.0
    tests['heavy0_wait2_amb1'] = lane0
    
    alt = np.zeros(runtime_input_size, dtype=float)
    for i in range(runtime_input_size):
        alt[i] = (i % 5) * 1.0
    tests['alt'] = alt

    from rl_model.environment import TrafficEnv

    def normalize_runtime_state(state, normalize_max, num_lanes, num_vehicle_types):
        # Use the single canonical normalizer so inspector matches training/GUI.
        return TrafficEnv.normalize_state_vector(
            np.array(state, dtype=float), num_lanes, num_vehicle_types, normalize_max
        )

    normalize_max = getattr(config, 'RL_NORMALIZE_MAX', 50.0)
    for name, st in tests.items():
        mapped = remap_runtime_to_model(st, model_input, num_lanes)
        norm = normalize_runtime_state(mapped, normalize_max, num_lanes, num_vehicle_types)
        print(f"Test {name}: runtime_len={len(st)} mapped_len={len(mapped)} mapped={np.array2string(mapped, precision=2)}")
        run_state(f"mapped_{name}", norm)

    print('Inspector done.')
