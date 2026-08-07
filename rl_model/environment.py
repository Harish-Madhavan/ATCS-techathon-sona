import numpy as np

def get_time_varying_arrival_rate(time_fraction, base_rate=0.1, peak_multiplier=4.0):
    """
    Simulates a rush hour pattern using a sinusoidal wave.
    time_fraction should be in [0, 1] representing progress through the episode.
    Starts at base_rate, peaks in the middle, and returns to base_rate.
    """
    # Scale fraction to be in the range [0, 2*pi]
    radian = time_fraction * 2 * np.pi
    # Use a sine wave shifted and scaled to create a single peak
    multiplier = 1 + (peak_multiplier - 1) * (-np.cos(radian) + 1) / 2
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
        self.max_time = 1800.0 # 30 mins of simulated time per episode
        self.current_time = 0.0
        self.current_step = 0

        if arrival_rate_per_lane is None:
            self.base_arrival_rate = 0.2
            arrival_rate_per_lane = [0.1, 0.4, 0.1, 0.4]
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
        self.consecutive_lane_count = 0

        # Pre-compute inverse-weight arrival probabilities so they aren't
        # recomputed on every single Poisson arrival step during simulation.
        weights_inv = 1.0 / (self.vehicle_weights + 1e-6)
        self._arrival_probs = weights_inv / np.sum(weights_inv)

    def get_one_hot_lane(self):
        one_hot = np.zeros(self.num_lanes, dtype=float)
        one_hot[self.current_lane] = 1.0
        return one_hot

    def normalize_state(self, state, normalize_max):
        # state is [vehicle_counts (24), wait_times (4), ambulance_flags (4), one_hot_lane (4)]
        v_end = self.num_lanes * (self.num_vehicle_types if self.num_vehicle_types > 0 else 1)
        w_end = v_end + self.num_lanes
        normalized = state.copy()
        normalized[:v_end] = normalized[:v_end] / max(1.0, float(normalize_max))
        # Use log1p scale for wait times so the agent can distinguish starvation
        # levels beyond 300 s. log(1+t)/log(301) maps [0, 300] -> [0, 1] and
        # continues to grow (slowly) for t > 300, avoiding the information loss
        # that a hard clamp introduces.
        _log_scale = np.log1p(300.0)  # ≈ 5.707
        normalized[v_end:w_end] = np.log1p(normalized[v_end:w_end]) / _log_scale
        # ambulance flags and one-hot lane remain as 0/1
        return normalized

    def reset(self):
        # random initial queues
        self.state_vehicle_counts = np.random.randint(0, 10, size=(self.num_lanes, self.num_vehicle_types))
        self.ambulance_presence = np.zeros(self.num_lanes, dtype=float)
        self.wait_times = np.zeros(self.num_lanes, dtype=float)
        self.current_lane = 0
        self.consecutive_lane_count = 0
        self.current_step = 0
        self.current_time = 0.0
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
            time_fraction = min(self.current_time / max(1.0, self.max_time), 1.0)
            current_arrival_rate = get_time_varying_arrival_rate(time_fraction, base_rate=self.base_arrival_rate)
            self.arrival_rate_per_lane = [current_arrival_rate] * self.num_lanes

        lane = int(action // self.num_duration_options)
        duration_index = int(action % self.num_duration_options)
        duration = float(self.duration_options[duration_index])
        
        # Track consecutive selections of the same lane
        if lane == self.current_lane:
            self.consecutive_lane_count += 1
        else:
            self.consecutive_lane_count = 1
            
        # Capture pre-service lane demands for priority bonus
        pre_demand = np.sum(self.state_vehicle_counts * self.vehicle_weights, axis=1)

        # Increment wait times for all lanes, then reset for the served lane
        for ln in range(self.num_lanes):
            if ln != lane:
                # Only accumulate wait time if there are vehicles waiting in this lane
                if np.sum(self.state_vehicle_counts[ln]) > 0:
                    self.wait_times[ln] += duration
                else:
                    self.wait_times[ln] = 0.0
            else:
                self.wait_times[ln] = 0.0

        # Check if ambulance was present in the served lane before resetting
        had_ambulance = (self.ambulance_presence[lane] > 0.0)
        self.ambulance_presence[lane] = 0.0

        # Service capacity: vehicles-per-second * duration -> capacity units in 'weight' units
        vehicle_flow_rate = 0.5  # vehicles per second base
        max_capacity_units = int(duration * vehicle_flow_rate)

        vehicles_in_lane = self.state_vehicle_counts[lane].copy()
        passed_total_units = 0
        vehicles_passed = np.zeros_like(vehicles_in_lane)

        # Serve heavier vehicles first (greedy choice)
        for i in np.argsort(self.vehicle_weights)[::-1]:
            if passed_total_units >= max_capacity_units:
                break
            available = int(vehicles_in_lane[i])
            capacity_left = max_capacity_units - passed_total_units
            can_pass = capacity_left // int(max(1, round(self.vehicle_weights[i])))
            num_pass = min(available, can_pass)
            vehicles_passed[i] = num_pass
            passed_total_units += int(num_pass * self.vehicle_weights[i])

        self.state_vehicle_counts[lane] = np.maximum(0, self.state_vehicle_counts[lane] - vehicles_passed)

        # Arrivals: Poisson arrivals per lane per vehicle type proportional to arrival_rate and duration
        for ln in range(self.num_lanes):
            lam = max(0.0, float(self.arrival_rate_per_lane[ln]) * max(1.0, duration))
            arrivals_total = np.random.poisson(lam)
            if arrivals_total > 0:
                arrivals_by_type = np.random.multinomial(arrivals_total, self._arrival_probs)
            else:
                arrivals_by_type = np.zeros(self.num_vehicle_types, dtype=int)
            self.state_vehicle_counts[ln] += arrivals_by_type

            # Simulate ambulance arrivals
            if np.random.random() < 0.02: # 2% chance per lane per decision step
                self.ambulance_presence[ln] = 1.0

        # Clamp queues
        self.state_vehicle_counts = np.minimum(self.state_vehicle_counts, self.max_queue)

        # Reward shaping components:
        cleared_value = float(np.sum(vehicles_passed * self.vehicle_weights))

        # Priority bonus: 0–2, proportional to how busy this lane was vs the busiest.
        priority_bonus = (float(pre_demand[lane]) / max(1.0, float(np.max(pre_demand)))) * 2.0

        # Normalized congestion penalty: split into queue size and quadratic wait times
        lane_queue_weights = np.sum(self.state_vehicle_counts * self.vehicle_weights, axis=1)
        queue_penalty = float(np.sum(lane_queue_weights)) * 0.5
        wait_penalty = float(np.sum((self.wait_times / 50.0) ** 2)) * 2.0
        congestion_penalty = queue_penalty + wait_penalty

        # Starvation penalty for lanes waiting too long (> 120s and > 200s)
        extreme_wait_penalty = 0.0
        for t in self.wait_times:
            if t > 200.0:
                extreme_wait_penalty += 80.0
            elif t > 120.0:
                extreme_wait_penalty += 40.0

        # Ambulance rewards/penalties
        ambulance_bonus = 30.0 if had_ambulance else 0.0
        # Dynamic accumulating penalty for letting ambulances wait
        ambulance_penalty = 0.0
        for ln in range(self.num_lanes):
            if self.ambulance_presence[ln] > 0.0:
                ambulance_penalty += 15.0 + 0.5 * self.wait_times[ln]

        # Wasted green light penalty (serving an empty lane)
        wasted_green_penalty = 0.0
        if pre_demand[lane] == 0.0 and not had_ambulance:
            wasted_green_penalty = 30.0

        # Consecutive lane selection penalty.
        # The training loop masks the same-lane action once consecutive_lane_count >= 2,
        # so count never exceeds 2 under a greedy policy — the old '> 2' branch was
        # dead code.  Replaced with an escalating signal: 1st repeat → 5, 2nd → 10
        # (consecutive_lane_count is already incremented at the top of this method).
        consecutive_penalty = 0.0
        if lane == self.current_lane:
            consecutive_penalty = 5.0 * self.consecutive_lane_count

        reward = (cleared_value + priority_bonus + ambulance_bonus - 
                  congestion_penalty - extreme_wait_penalty - ambulance_penalty - 
                  wasted_green_penalty - consecutive_penalty) / 100.0

        self.current_lane = lane
        self.current_time += duration
        done = self.current_time >= self.max_time
        next_state = self._get_state()

        # Pack detailed metrics for benchmarking
        info = {
            "cleared_value": cleared_value,
            "priority_bonus": priority_bonus,
            "ambulance_bonus": ambulance_bonus,
            "congestion_penalty": congestion_penalty,
            "extreme_wait_penalty": extreme_wait_penalty,
            "ambulance_penalty": ambulance_penalty,
            "wasted_green_penalty": wasted_green_penalty,
            "consecutive_penalty": consecutive_penalty,
            "vehicles_passed": float(np.sum(vehicles_passed))
        }

        return next_state, reward, done, info
