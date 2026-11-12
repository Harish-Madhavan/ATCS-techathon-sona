from collections import defaultdict
import numpy as np

from core.intersection_state import IntersectionState
from core.log import get_logger

logger = get_logger(__name__)

class TrafficLightController:
    def __init__(self, config_data, vehicle_type_weights=None, default_vehicle_weight=1.0):
        self.intersections = {}
        self.config = config_data
        self.all_approach_names = set()
        self.vehicle_type_weights = vehicle_type_weights if vehicle_type_weights is not None else {}
        self.default_vehicle_weight = default_vehicle_weight
        # Deterministic ordering of vehicle types used for RL state construction
        from core import config
        self.vehicle_type_order = getattr(config, 'TARGET_CLASSES', list(self.vehicle_type_weights.keys()))

        if not config_data:
            logger.error("No configuration data provided.")
            raise ValueError("Traffic light configuration cannot be empty.")

        logger.debug("Initializing intersections...")
        for name, int_config in config_data.items():
            try:
                self._initialize_intersection_state(name, int_config)
            except ValueError as e:
                 logger.error("Failed to initialize intersection '%s': %s", name, e)
                 raise

        if not self.intersections:
             logger.error("No intersections were successfully initialized.")
             raise ValueError("No valid intersections configured.")

        logger.debug(
            "Controller initialized for %s; approaches=%s",
            list(self.intersections.keys()), sorted(list(self.all_approach_names)),
        )


    def _validate_phase_config(self, name, phases_config):
        if not phases_config or not isinstance(phases_config, dict):
            raise ValueError(f"'{name}': 'phases' dictionary is missing or invalid.")
        all_phase_approaches = set()
        for phase_name, approaches in phases_config.items():
            if not isinstance(approaches, list):
                raise ValueError(f"'{name}', phase '{phase_name}': Approaches must be a list.")
            if len(approaches) != 1:
                raise ValueError(
                    f"Intersection '{name}', phase '{phase_name}': Must contain exactly one approach name "
                    f"for single-approach green logic. Found {len(approaches)}: {approaches}. "
                    "Please ensure each phase in TRAFFIC_LIGHT_CONFIG maps to a list with a single approach string."
                )
            all_phase_approaches.update(approaches)
        return all_phase_approaches

    def _initialize_intersection_state(self, name, config):
        phases_config = config.get('phases', {})
        intersection_approaches = self._validate_phase_config(name, phases_config)
        phase_names_list = list(phases_config.keys())
        if not phase_names_list:
             raise ValueError(f"No phase names found for intersection '{name}'.")

        self.all_approach_names.update(intersection_approaches)
        timings = config.get('timings', {})
        required_keys = ['min_green', 'yellow', 'all_red', 'gap_time', 'skip_threshold',
                         'emergency_green', 'ambulance_request_timeout',
                         'base_max_green', 'queued_weighted_demand_extension_factor', 'absolute_max_green',
                         'realtime_flow_extension_increment', 'realtime_flow_min_weighted_demand'
                         ]
        missing_keys = [k for k in required_keys if k not in timings]
        if missing_keys:
             raise ValueError(f"Missing timing keys in config for intersection '{name}': {missing_keys}")

        demand_threshold = config.get('demand_threshold', 1)
        if not isinstance(demand_threshold, (int, float)) or demand_threshold < 0:
             logger.warning("Invalid demand_threshold for '%s'. Using default 1.", name)
             demand_threshold = 1

        self.intersections[name] = IntersectionState(
            name=name,
            config=config,
            phases=phase_names_list,
            intersection_approaches=intersection_approaches,
            timings=timings,
            demand_threshold=demand_threshold,
            vehicle_type_order=self.vehicle_type_order
        )
        logger.debug("Initialized '%s': ALL_RED, first phase '%s'.", name, phase_names_list[0])

    
    def set_manual_override(self, intersection_name, approach_name, is_forced_red):
        if intersection_name in self.intersections:
            state = self.intersections[intersection_name]
            if approach_name in state["managed_approaches"]:
                state["manual_override_red"][approach_name] = is_forced_red
                action = "FORCED RED" if is_forced_red else "RELEASED from manual red"
                logger.info("[%s] Manual override %s: %s", state['name'], approach_name, action)

                
                
                return True
        logger.warning("Could not set manual override for '%s' in '%s'.", approach_name, intersection_name)
        return False
    

    def update_demand(self, approach_name, count, current_time, ambulance_detected=False):
        for int_state in self.intersections.values():
            if approach_name in int_state["managed_approaches"]:
                current_phase_key = int_state['phases'][int_state['current_phase_index']]
                current_phase_approaches = int_state['config']['phases'].get(current_phase_key, [])
                is_in_active_phase = (approach_name in current_phase_approaches)
                is_green_or_yellow = is_in_active_phase and (int_state['current_state'] in ["GREEN", "YELLOW"])

                if is_green_or_yellow:
                    if count > 0:
                        int_state['last_detection_time_green'][approach_name] = current_time
                    int_state['approach_demand'][approach_name] = 0
                else:
                    int_state['approach_demand'][approach_name] = count
                
                if ambulance_detected:
                    int_state['ambulance_request_active'][approach_name] = True
                    int_state['last_ambulance_detection_time'][approach_name] = current_time
                return

    def update_weighted_demand(self, approach_name, counts_by_type, current_time):
        current_weighted_value_this_update = 0
        lower_weight_map = {k.lower(): v for k, v in self.vehicle_type_weights.items()}
        for vehicle_type, count in counts_by_type.items():
            weight = lower_weight_map.get(str(vehicle_type).lower(), self.default_vehicle_weight)
            current_weighted_value_this_update += count * weight
        
        for int_state in self.intersections.values():
            if approach_name in int_state["managed_approaches"]:
                # Store the latest per-vehicle-type counts in a deterministic order
                if hasattr(int_state, 'get'):
                    pass
                try:
                    # Ensure all known vehicle types present; unknown types ignored
                    stored = {vt: int(counts_by_type.get(vt, 0)) for vt in self.vehicle_type_order}
                    int_state['approach_counts_by_type'][approach_name] = stored
                except Exception:
                    # If vehicle_type_order is empty or something fails, skip storing counts
                    pass
                current_phase_key = int_state['phases'][int_state['current_phase_index']]
                current_phase_approaches = int_state['config']['phases'].get(current_phase_key, [])
                is_in_active_phase = (approach_name in current_phase_approaches)
                is_green = is_in_active_phase and (int_state['current_state'] == "GREEN")

                if is_green:
                    int_state['last_weighted_flow_green'][approach_name] = max(
                        int_state['last_weighted_flow_green'].get(approach_name, 0.0),
                        current_weighted_value_this_update
                    )
                    int_state['approach_weighted_demand'][approach_name] = 0.0
                elif is_in_active_phase and int_state['current_state'] == "YELLOW":
                    int_state['approach_weighted_demand'][approach_name] = 0.0
                else:
                    int_state['approach_weighted_demand'][approach_name] = current_weighted_value_this_update
                return

    def update_state(self, current_time, rl_action=None):
        any_state_changed = False
        for name, state in self.intersections.items():
            try:
                if state.update_single_intersection_state(current_time, rl_action=rl_action):
                    any_state_changed = True
            except Exception as e:
                logger.exception("Unhandled exception in update_single_intersection_state for '%s': %s", name, e)

    def get_rl_state(self, intersection_name):
        state_data = self.intersections.get(intersection_name)
        if not state_data:
            return None
        # Use deterministic phase order assigned at initialization
        phase_order = state_data.get('rl_phase_order', state_data['phases'])
        num_lanes = len(phase_order)

        # Build vehicle counts per lane in the configured vehicle type order
        vt_order = self.vehicle_type_order if self.vehicle_type_order else []
        counts_list = []
        current_phase_key = state_data['phases'][state_data['current_phase_index']]
        current_state = state_data['current_state']

        for phase_key in phase_order:
            approach_name = state_data['config']['phases'][phase_key][0]
            # If approach is in GREEN or YELLOW phase, queued demand is zero (vehicles passing through)
            is_active_green_or_yellow = (phase_key == current_phase_key) and (current_state in ["GREEN", "YELLOW"])
            if is_active_green_or_yellow:
                stored_counts = {}
            else:
                stored_counts = state_data['approach_counts_by_type'].get(approach_name, {})

            # If no vehicle types configured, fall back to weighted demand as single scalar
            if not vt_order:
                demand = 0.0 if is_active_green_or_yellow else state_data['approach_weighted_demand'].get(approach_name, 0.0)
                counts_list.append([demand])
            else:
                per_type = [float(stored_counts.get(vt, 0)) for vt in vt_order]
                counts_list.append(per_type)

        vehicle_counts = np.array(counts_list, dtype=float)

        # Add wait times (normalized by time since last green or current wait metrics)
        wait_times = []
        for phase_key in phase_order:
            approach_name = state_data['config']['phases'][phase_key][0]
            # Use approach_wait_time if available, else 0
            w = float(state_data['approach_wait_time'].get(approach_name, 0.0))
            wait_times.append(w)
        wait_times = np.array(wait_times, dtype=float)

        # Add ambulance flags
        ambulance_flags = []
        for phase_key in phase_order:
            approach_name = state_data['config']['phases'][phase_key][0]
            is_amb = 1.0 if state_data['ambulance_request_active'].get(approach_name, False) else 0.0
            ambulance_flags.append(is_amb)
        ambulance_flags = np.array(ambulance_flags, dtype=float)

        # one-hot of the current phase in the same phase order
        one_hot_lane = np.zeros(num_lanes)
        current_phase_key = state_data['phases'][state_data['current_phase_index']]
        if current_phase_key in phase_order:
            current_idx = phase_order.index(current_phase_key)
            one_hot_lane[current_idx] = 1

        return np.concatenate([vehicle_counts.flatten(), wait_times, ambulance_flags, one_hot_lane])

    def get_intersection_names(self):
        return list(self.intersections.keys())

    def get_approaches_for_intersection(self, intersection_name):
        state = self.intersections.get(intersection_name)
        return state["managed_approaches"] if state else []

    def get_all_approach_names(self):
        return sorted(list(self.all_approach_names))

    def get_intersection_status(self, intersection_name):
        state = self.intersections.get(intersection_name)
        if not state: return {}
        
        current_phase_key = state['phases'][state['current_phase_index']]
        approach_for_current_phase_list = state['config']['phases'].get(current_phase_key, [])
        approach_for_current_phase = "N/A"
        is_current_phase_manually_red = False
        if approach_for_current_phase_list:
            approach_for_current_phase = approach_for_current_phase_list[0]
            is_current_phase_manually_red = state["manual_override_red"].get(approach_for_current_phase, False)
        
        timer_val = state['green_timer'] if state['current_state'] == 'GREEN' else state['state_timer']
        
        max_duration_for_progress = 0
        current_s = state['current_state']
        timings = state['config']['timings']

        if current_s == 'GREEN':
            
            
            max_duration_for_progress = state['current_cycle_max_green'] if not state['is_current_phase_emergency'] else timings['emergency_green']
            if is_current_phase_manually_red : 
                 max_duration_for_progress = 0.1 
        elif current_s == 'YELLOW':
            max_duration_for_progress = timings['yellow']
        elif current_s == 'ALL_RED':
            max_duration_for_progress = timings['all_red']
        
        return {
            'phase': current_phase_key,
            'active_approach': approach_for_current_phase, 
            'state': current_s, 
            'timer': timer_val,
            'max_duration': max_duration_for_progress,
            'is_emergency': state['is_current_phase_emergency'] or state['emergency_preemption_active']
        }

    def get_all_approach_statuses(self):
        statuses = {}
        for int_name, int_state in self.intersections.items():
            current_phase_key = int_state['phases'][int_state['current_phase_index']]
            current_green_approach_list = int_state['config']['phases'].get(current_phase_key, [])
            current_intersection_actual_state = int_state['current_state']

            for approach_name in int_state["managed_approaches"]:
                light_state_for_approach = 'RED' 
                
                
                is_manually_forced_red = int_state["manual_override_red"].get(approach_name, False)
                if is_manually_forced_red:
                    light_state_for_approach = 'RED'
                elif current_intersection_actual_state in ['GREEN', 'YELLOW']:
                    
                    if current_green_approach_list and approach_name == current_green_approach_list[0]:
                        light_state_for_approach = current_intersection_actual_state
                
                
                statuses[approach_name] = {
                    'state': light_state_for_approach,
                    'demand': int_state['approach_demand'].get(approach_name, 0),
                    'weighted_demand': int_state['approach_weighted_demand'].get(approach_name, 0.0),
                    'ambulance_request_active': int_state['ambulance_request_active'].get(approach_name, False),
                    'is_manually_red': is_manually_forced_red 
                }
        return statuses