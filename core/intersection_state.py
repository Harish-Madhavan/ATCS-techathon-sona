import time
from collections import defaultdict

class IntersectionState(dict):
    def __init__(self, name, config, phases, intersection_approaches, timings, demand_threshold, vehicle_type_order):
        super().__init__()
        self["config"] = config
        self["phases"] = phases
        self["current_phase_index"] = 0
        self["current_state"] = "ALL_RED"
        self["state_timer"] = 0.0
        self["green_timer"] = 0.0
        self["current_cycle_max_green"] = timings.get('base_max_green', 20)
        self["last_update_time"] = time.time()
        self["approach_demand"] = defaultdict(int)
        self["approach_weighted_demand"] = defaultdict(float)
        self["approach_wait_time"] = defaultdict(float)
        self["approach_counts_by_type"] = defaultdict(lambda: {vt: 0 for vt in vehicle_type_order})
        self["last_detection_time_green"] = defaultdict(float)
        self["last_weighted_flow_green"] = defaultdict(float)
        self["rl_enforced"] = False
        self["name"] = name
        self["managed_approaches"] = list(intersection_approaches)
        self["demand_threshold"] = demand_threshold
        self["ambulance_request_active"] = defaultdict(bool)
        self["last_ambulance_detection_time"] = defaultdict(float)
        self["emergency_preemption_active"] = False
        self["target_emergency_phase_key"] = None
        self["is_current_phase_emergency"] = False
        self["manual_override_red"] = defaultdict(bool) 
        self["rl_phase_order"] = phases
        self["last_rl_chosen_lane"] = None
        self["consecutive_rl_same_lane"] = 0

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def _update_ambulance_request_timeouts(self, current_time):
        timeout_duration = self['config']['timings']['ambulance_request_timeout']
        for approach_name in list(self['ambulance_request_active'].keys()):
            if self['ambulance_request_active'][approach_name]:
                last_det_time = self['last_ambulance_detection_time'].get(approach_name, 0)
                if current_time - last_det_time > timeout_duration:
                    print(f"[{self['name']}] Ambulance request for {approach_name} timed out.")
                    self['ambulance_request_active'][approach_name] = False

    def _check_for_emergency_preemption_need(self):
        if self['emergency_preemption_active'] and self['target_emergency_phase_key']:
            return None 
        phases_config = self['config']['phases']
        phase_keys_list = self['phases']
        for p_key in phase_keys_list: 
            phase_approaches = phases_config.get(p_key, []) 
            if not phase_approaches: continue
            the_single_approach_for_this_phase = phase_approaches[0]

            if self["manual_override_red"].get(the_single_approach_for_this_phase, False):
                continue

            if self['ambulance_request_active'].get(the_single_approach_for_this_phase, False):
                is_current_green_for_this_amb_phase = (
                    self['current_state'] == "GREEN" and
                    phase_keys_list[self['current_phase_index']] == p_key and
                    self['is_current_phase_emergency']
                )
                if not is_current_green_for_this_amb_phase:
                    return p_key 
        return None

    def _update_timers(self, current_time):
        delta_time = current_time - self['last_update_time']
        max_delta = 5.0
        if delta_time < 0:
            delta_time = 0.0
        if delta_time > max_delta:
            print(f"[{self['name']}] Warning: Large time delta ({delta_time:.1f}s). Clamping to {max_delta}s.")
            delta_time = max_delta
            
        self['state_timer'] += delta_time
        if self['current_state'] == "GREEN":
            self['green_timer'] += delta_time
            try:
                print(f"[{self['name']}] DEBUG: green_timer={self['green_timer']:.1f} max_green={self.get('current_cycle_max_green',0):.1f} rl_enforced={self.get('rl_enforced',False)}")
            except Exception:
                pass
        return delta_time

    def _update_wait_times(self, delta_time, current_green_approach):
        for approach in self['managed_approaches']:
            if self['current_state'] == "GREEN" and approach == current_green_approach:
                self['approach_wait_time'][approach] = 0.0
            else:
                if self['approach_demand'].get(approach, 0) > 0:
                    self['approach_wait_time'][approach] += delta_time
                else:
                    self['approach_wait_time'][approach] = 0.0

    def _handle_green_transitions(self, timings, phases_config, phase_keys_list, green_approach, current_time):
        current_phase_key = phase_keys_list[self['current_phase_index']]

        if green_approach and self["manual_override_red"].get(green_approach, False):
            return "YELLOW", f"Current green approach '{green_approach}' MANUALLY FORCED TO RED."

        if self['is_current_phase_emergency']:
            if self['green_timer'] >= timings['emergency_green']:
                return "YELLOW", f"Emergency Green ({timings['emergency_green']}s) for '{current_phase_key}' ({green_approach}) finished."

        if not self['is_current_phase_emergency']:
            # Extension check
            try:
                if self['current_cycle_max_green'] < timings['absolute_max_green'] and self['last_weighted_flow_green'].get(green_approach, 0.0) >= timings['realtime_flow_min_weighted_demand']:
                    new_max_green = min(self['current_cycle_max_green'] + timings['realtime_flow_extension_increment'], timings['absolute_max_green'])
                    if new_max_green > self['current_cycle_max_green']:
                        print(f"[{self['name']}] Extension: Approach '{green_approach}' extending max green {self['current_cycle_max_green']:.1f}s -> {new_max_green:.1f}s due to realtime flow {self['last_weighted_flow_green'].get(green_approach,0.0):.1f}")
                        self['current_cycle_max_green'] = new_max_green
                    self['last_weighted_flow_green'][green_approach] = 0.0
            except Exception:
                pass

            # Timeout or Gapout check
            if self['green_timer'] >= self['current_cycle_max_green']:
                return "YELLOW", f"Calculated Max Green ({self['current_cycle_max_green']:.1f}s) reached for {green_approach}"
            
            elif self['green_timer'] >= timings['min_green'] and not self.get('rl_enforced', False):
                max_conflicting_weighted_demand = 0.0
                conflicting_approach_display_name = "N/A"
                for other_phase_idx, other_p_key in enumerate(phase_keys_list):
                    if other_phase_idx != self['current_phase_index']:
                        approaches_in_other_phase = phases_config.get(other_p_key, [])
                        if not approaches_in_other_phase:
                            continue
                        the_other_approach = approaches_in_other_phase[0]
                        if self["manual_override_red"].get(the_other_approach, False):
                            continue
                        demand_on_other_approach = self['approach_weighted_demand'].get(the_other_approach, 0.0)
                        if demand_on_other_approach > max_conflicting_weighted_demand:
                            max_conflicting_weighted_demand = demand_on_other_approach
                            conflicting_approach_display_name = the_other_approach

                conflicting_demand_met = (max_conflicting_weighted_demand >= self['demand_threshold'])
                last_green_det_time = self['last_detection_time_green'].get(green_approach, 0.0)
                time_since_last_green = current_time - last_green_det_time if last_green_det_time > 0 else timings['gap_time'] + 1

                if conflicting_demand_met and time_since_last_green > timings['gap_time']:
                    return "YELLOW", (f"Gap-Out on '{green_approach}' (Gap: {time_since_last_green:.1f}s > {timings['gap_time']}s | "
                                     f"Conflict: Approach '{conflicting_approach_display_name}' W.Demand={max_conflicting_weighted_demand:.1f})")
        return None, ""

    def _handle_yellow_transitions(self, timings, green_approach):
        if self['state_timer'] >= timings['yellow']:
            return "ALL_RED", f"Yellow time for '{green_approach}' finished"
        return None, ""

    def _handle_all_red_transitions(self, current_time, timings, phases_config, phase_keys_list, rl_action):
        if self['state_timer'] < timings['all_red']:
            return None, ""

        switch_reason_prefix = "All Red finished. "
        
        # 1. Check for emergency preemptions first
        if self['emergency_preemption_active'] and self['target_emergency_phase_key']:
            try:
                target_emergency_approach_name = phases_config.get(self['target_emergency_phase_key'], ["Unknown"])[0]
                if self["manual_override_red"].get(target_emergency_approach_name, False):
                    print(f"[{self['name']}] Emergency target '{target_emergency_approach_name}' (Phase '{self['target_emergency_phase_key']}') is MANUALLY FORCED RED. Cannot service emergency.")
                    self['emergency_preemption_active'] = False 
                    self['target_emergency_phase_key'] = None
                    
                    new_emergency_target = self._check_for_emergency_preemption_need()
                    if new_emergency_target:
                        self['emergency_preemption_active'] = True
                        self['target_emergency_phase_key'] = new_emergency_target
                else: 
                    self['current_phase_index'] = phase_keys_list.index(self['target_emergency_phase_key'])
                    self['is_current_phase_emergency'] = True 
                    
                    emergency_approaches_served = phases_config.get(self['target_emergency_phase_key'], [])
                    if emergency_approaches_served:
                        served_approach_name = emergency_approaches_served[0]
                        if self['ambulance_request_active'].get(served_approach_name, False):
                            print(f"[{self['name']}] Servicing ambulance for {served_approach_name} on phase '{self['target_emergency_phase_key']}'. Clearing request.")
                            self['ambulance_request_active'][served_approach_name] = False
                    
                    return "GREEN", switch_reason_prefix + f"Starting EMERGENCY Phase '{self['target_emergency_phase_key']}' for approach '{target_emergency_approach_name}'"
            except ValueError:
                print(f"[{self['name']}] ERROR: Target emergency phase '{self['target_emergency_phase_key']}' not found. Reverting.")
                self['emergency_preemption_active'] = False
                self['target_emergency_phase_key'] = None
                self['is_current_phase_emergency'] = False

        # 2. RL phase selection
        if rl_action is not None:
            try:
                from core import config as cfg
                duration_options = getattr(cfg, 'DURATION_OPTIONS', [10, 20, 30])
                num_duration_options = len(duration_options)
                if num_duration_options <= 0:
                    raise ValueError("No duration options configured for RL decoding.")

                chosen_lane_index = rl_action // num_duration_options
                chosen_duration_index = rl_action % num_duration_options

                phase_keys_list_ordered = self.get('rl_phase_order', phase_keys_list)
                if chosen_lane_index < 0 or chosen_lane_index >= len(phase_keys_list_ordered):
                    raise IndexError(f"Decoded lane index {chosen_lane_index} out of range (0..{len(phase_keys_list_ordered)-1}).")
                if chosen_duration_index < 0 or chosen_duration_index >= num_duration_options:
                    raise IndexError(f"Decoded duration index {chosen_duration_index} out of range (0..{num_duration_options-1}).")

                chosen_duration = duration_options[chosen_duration_index]
                next_phase_key_selected = phase_keys_list_ordered[chosen_lane_index]

                try:
                    from core import config as cfg
                    max_repeat = getattr(cfg, 'MAX_CONSECUTIVE_RL_SAME_LANE', 2)
                except Exception:
                    max_repeat = 2

                last_rl_lane = self.get('last_rl_chosen_lane')
                chosen_lane_name = next_phase_key_selected
                if last_rl_lane == chosen_lane_name and self.get('consecutive_rl_same_lane', 0) >= max_repeat:
                    found_alternative = False
                    for offset in range(1, len(phase_keys_list_ordered)):
                        alt_idx = (chosen_lane_index + offset) % len(phase_keys_list_ordered)
                        alt_phase = phase_keys_list_ordered[alt_idx]
                        alt_approach = phases_config.get(alt_phase, [None])[0]
                        if alt_approach and not self['manual_override_red'].get(alt_approach, False):
                            next_phase_key_selected = alt_phase
                            chosen_lane_index = alt_idx
                            found_alternative = True
                            break

                self['current_phase_index'] = phase_keys_list.index(next_phase_key_selected)
                selected_approach_for_green_candidate = phases_config.get(next_phase_key_selected, ["Unknown"])[0]

                calc_g = float(chosen_duration)
                calc_g = min(calc_g, timings.get('absolute_max_green', calc_g))
                calc_g = max(calc_g, timings.get('min_green', calc_g))
                self['current_cycle_max_green'] = calc_g
                self['rl_enforced'] = True
                
                prev = self.get('last_rl_chosen_lane')
                if prev == next_phase_key_selected:
                    self['consecutive_rl_same_lane'] = self.get('consecutive_rl_same_lane', 0) + 1
                else:
                    self['consecutive_rl_same_lane'] = 1
                self['last_rl_chosen_lane'] = next_phase_key_selected
                print(f"[{self['name']}] RL set rl_enforced=True for '{selected_approach_for_green_candidate}'")
                print(f"[{self['name']}] RL set green for '{selected_approach_for_green_candidate}' to {self['current_cycle_max_green']:.1f}s (chosen_duration={chosen_duration})")

                self['is_current_phase_emergency'] = False
                
                return "GREEN", switch_reason_prefix + f"RL CHOSE Phase '{next_phase_key_selected}' for approach '{selected_approach_for_green_candidate}' with duration {self['current_cycle_max_green']:.1f}s"
            except Exception as e_rl_decode:
                print(f"[{self['name']}] Error decoding RL action: {e_rl_decode}")
                rl_action = None

        # 3. Fallback round-robin selection
        print(f"[{self['name']}] No RL action provided; falling back to round-robin.")
        next_phase_idx = (self['current_phase_index'] + 1) % len(phase_keys_list)
        self['current_phase_index'] = next_phase_idx
        selected_approach = phases_config.get(phase_keys_list[next_phase_idx], ["Unknown"])[0]
        self['current_cycle_max_green'] = timings.get('base_max_green', 20)
        self['is_current_phase_emergency'] = False

        if self['emergency_preemption_active'] and phase_keys_list[self['current_phase_index']] != self['target_emergency_phase_key']:
             print(f"[{self['name']}] Emergency preemption for '{self['target_emergency_phase_key']}' concluded as normal phase '{phase_keys_list[self['current_phase_index']]}' starts.")
             self['emergency_preemption_active'] = False
             self['target_emergency_phase_key'] = None

        return "GREEN", switch_reason_prefix + f"Fallback round-robin selected Phase '{phase_keys_list[next_phase_idx]}' for approach '{selected_approach}'"

    def _apply_state_change(self, current_time, next_state, switch_reason, timings, phases_config, phase_keys_list, current_phase_key_on_entry):
        current_phase_key_display = phase_keys_list[self['current_phase_index']]
        current_approach_display = phases_config.get(current_phase_key_display, ["Unknown"])[0]
        print(f"[{self['name']}] State Change: {self['current_state']} -> {next_state}. (Phase: {current_phase_key_display} for Appr: {current_approach_display}, Reason: {switch_reason})")
        
        if self['current_state'] == "YELLOW" and next_state == "ALL_RED":
             finished_approaches = phases_config.get(current_phase_key_on_entry, []) 
             if finished_approaches:
                 the_finished_approach = finished_approaches[0]
                 
                 if not self["manual_override_red"].get(the_finished_approach, False):
                      self['approach_demand'][the_finished_approach] = 0 
                      self['approach_weighted_demand'][the_finished_approach] = 0.0
                 if the_finished_approach in self['last_detection_time_green']:
                      del self['last_detection_time_green'][the_finished_approach]
                 if the_finished_approach in self['last_weighted_flow_green']:
                      del self['last_weighted_flow_green'][the_finished_approach]
             
             if self['is_current_phase_emergency']:
                  print(f"[{self['name']}] Emergency phase '{current_phase_key_on_entry}' for approach '{current_approach_display}' cycle completed.")
                  self['is_current_phase_emergency'] = False 
                  if self['emergency_preemption_active'] and self['target_emergency_phase_key'] == current_phase_key_on_entry:
                      another_emergency = self._check_for_emergency_preemption_need()
                      if not another_emergency:
                          print(f"[{self['name']}] No further pending emergencies. Deactivating preemption mode.")
                          self['emergency_preemption_active'] = False
                          self['target_emergency_phase_key'] = None
                      else:
                          print(f"[{self['name']}] Another emergency for '{another_emergency}' detected. Preemption remains active.")
                          self['target_emergency_phase_key'] = another_emergency 

        self['current_state'] = next_state
        self['state_timer'] = 0.0
        if next_state == "GREEN":
            self['green_timer'] = 0.0
            new_green_approach_key = phase_keys_list[self['current_phase_index']]
            new_green_approach = phases_config.get(new_green_approach_key, ["Unknown"])[0]
            if new_green_approach != "Unknown":
                self['last_weighted_flow_green'][new_green_approach] = 0.0
        else:
            self['rl_enforced'] = False

    def update_single_intersection_state(self, current_time, rl_action=None):
        config = self['config']
        timings = config['timings']
        phases_config = config['phases']
        phase_keys_list = self['phases']
        
        # 1. Update timers
        delta_time = self._update_timers(current_time)

        # 2. Update ambulance timeouts
        self._update_ambulance_request_timeouts(current_time)

        # 3. Check for emergency preemptions
        next_state = None
        switch_reason = ""
        state_did_change = False

        if not self['emergency_preemption_active']: 
            emergency_phase_needed = self._check_for_emergency_preemption_need()
            if emergency_phase_needed:
                self['emergency_preemption_active'] = True
                self['target_emergency_phase_key'] = emergency_phase_needed
                self['is_current_phase_emergency'] = False 
                print(f"[{self['name']}] EMERGENCY PREEMPTION ACTIVATED for phase '{emergency_phase_needed}'.")
                if self['current_state'] == "GREEN" and phase_keys_list[self['current_phase_index']] != emergency_phase_needed:
                    next_state = "YELLOW"
                    switch_reason = f"Emergency Preemption for '{emergency_phase_needed}'"

        current_phase_key_on_entry = phase_keys_list[self['current_phase_index']]
        current_phase_actual_approaches_on_entry = phases_config.get(current_phase_key_on_entry, [])
        the_current_green_approach_on_entry = current_phase_actual_approaches_on_entry[0] if current_phase_actual_approaches_on_entry else None

        # 4. Update wait times
        self._update_wait_times(delta_time, the_current_green_approach_on_entry)

        # 5. Process state machine triggers
        if next_state is None:
            next_state, switch_reason = self._handle_transitions_by_state(current_time, timings, phases_config, phase_keys_list, the_current_green_approach_on_entry, rl_action)

        # 6. Apply state change if triggered
        if next_state and next_state != self['current_state']:
            self._apply_state_change(current_time, next_state, switch_reason, timings, phases_config, phase_keys_list, current_phase_key_on_entry)
            state_did_change = True
        
        self['last_update_time'] = current_time
        return state_did_change

    def _handle_transitions_by_state(self, current_time, timings, phases_config, phase_keys_list, green_approach, rl_action):
        current_state = self['current_state']
        if current_state == "GREEN":
            return self._handle_green_transitions(timings, phases_config, phase_keys_list, green_approach, current_time)
        elif current_state == "YELLOW":
            return self._handle_yellow_transitions(timings, green_approach)
        elif current_state == "ALL_RED":
            return self._handle_all_red_transitions(current_time, timings, phases_config, phase_keys_list, rl_action)
        return None, ""
