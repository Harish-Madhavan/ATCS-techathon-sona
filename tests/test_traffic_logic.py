import unittest
import numpy as np
import time

from core import config
from core.traffic_logic import TrafficLightController


class TestTrafficLogic(unittest.TestCase):
    def setUp(self):
        self.test_config = {
            "Intersection1": {
                "phases": {
                    "GreenForNorth": ["Northbound"],
                    "GreenForEast": ["Eastbound"],
                    "GreenForWest": ["Westbound"],
                    "GreenForSouth": ["Southbound"]
                },
                "timings": {
                    "min_green": 5,
                    "yellow": 2,
                    "all_red": 1,
                    "gap_time": 3.0,
                    "skip_threshold": 2.0,
                    "emergency_green": 8,
                    "ambulance_request_timeout": 5.0,
                    "base_max_green": 15,
                    "queued_weighted_demand_extension_factor": 0.5,
                    "absolute_max_green": 30,
                    "realtime_flow_extension_increment": 2.0,
                    "realtime_flow_min_weighted_demand": 2.0,
                },
                "demand_threshold": 2.0
            }
        }
        self.weights = {'car': 1.0, 'bus': 3.0, 'truck': 2.0}
        self.controller = TrafficLightController(
            self.test_config,
            vehicle_type_weights=self.weights,
            default_vehicle_weight=1.0
        )
        self.int_name = "Intersection1"
        self.state = self.controller.intersections[self.int_name]
        # Reset last_update_time to 0.0 for deterministic simulation stepping
        self.state['last_update_time'] = 0.0

    def test_controller_initialization(self):
        """Verify controller properly registers approaches and initial state."""
        approaches = self.controller.get_all_approach_names()
        self.assertEqual(sorted(approaches), ["Eastbound", "Northbound", "Southbound", "Westbound"])
        status = self.controller.get_intersection_status(self.int_name)
        self.assertEqual(status['state'], 'ALL_RED')

    def test_fsm_transition_cycle(self):
        """Verify full FSM cycle: ALL_RED -> GREEN -> YELLOW -> ALL_RED."""
        t = 0.0
        # 1. Starting in ALL_RED at t=0
        self.controller.update_state(t)
        self.assertEqual(self.state['current_state'], 'ALL_RED')

        # Advance past all_red (1s)
        t = 1.5
        self.controller.update_state(t)
        self.assertEqual(self.state['current_state'], 'GREEN')
        active_approach = self.state['config']['phases'][self.state['phases'][self.state['current_phase_index']]][0]
        self.assertEqual(active_approach, 'Eastbound')

        # Advance while in GREEN until transition to YELLOW (due to max green)
        while self.state['current_state'] == 'GREEN':
            t += 1.0
            self.controller.update_state(t)
        self.assertEqual(self.state['current_state'], 'YELLOW')

        # Advance while in YELLOW until transition to ALL_RED
        while self.state['current_state'] == 'YELLOW':
            t += 1.0
            self.controller.update_state(t)
        self.assertEqual(self.state['current_state'], 'ALL_RED')

        # Advance while in ALL_RED until transition to GREEN (next phase: Westbound)
        while self.state['current_state'] == 'ALL_RED':
            t += 1.0
            self.controller.update_state(t)
        self.assertEqual(self.state['current_state'], 'GREEN')
        next_approach = self.state['config']['phases'][self.state['phases'][self.state['current_phase_index']]][0]
        self.assertEqual(next_approach, 'Westbound')

    def test_ambulance_emergency_preemption(self):
        """Verify ambulance detection preempts current green phase."""
        # Advance past all_red (1s) to enter GREEN
        t = 1.5
        self.controller.update_state(t)
        self.assertEqual(self.state['current_state'], 'GREEN')

        # Ambulance detected on Westbound
        self.controller.update_demand('Westbound', count=1, current_time=t, ambulance_detected=True)
        self.controller.update_state(t)

        # Should transition immediately to YELLOW to preempt
        self.assertEqual(self.state['current_state'], 'YELLOW')
        self.assertTrue(self.state['emergency_preemption_active'])
        self.assertEqual(self.state['target_emergency_phase_key'], 'GreenForWest')

        # Finish yellow (2s)
        t += 2.5
        self.controller.update_state(t)
        self.assertEqual(self.state['current_state'], 'ALL_RED')

        # Finish all_red (1s) -> GREEN for Westbound emergency
        t += 1.5
        self.controller.update_state(t)
        self.assertEqual(self.state['current_state'], 'GREEN')
        self.assertTrue(self.state['is_current_phase_emergency'])
        curr_phase = self.state['phases'][self.state['current_phase_index']]
        self.assertEqual(curr_phase, 'GreenForWest')

    def test_manual_override_red(self):
        """Verify manual override red forces approach to red and skips in phase selection."""
        # Force Northbound to red
        success = self.controller.set_manual_override(self.int_name, 'Northbound', is_forced_red=True)
        self.assertTrue(success)

        statuses = self.controller.get_all_approach_statuses()
        self.assertTrue(statuses['Northbound']['is_manually_red'])
        self.assertEqual(statuses['Northbound']['state'], 'RED')

        # Release override
        self.controller.set_manual_override(self.int_name, 'Northbound', is_forced_red=False)
        statuses = self.controller.get_all_approach_statuses()
        self.assertFalse(statuses['Northbound']['is_manually_red'])

    def test_rl_state_vector(self):
        """Verify RL state vector layout and dimensionality."""
        t = 10.0
        self.controller.update_weighted_demand('Northbound', {'car': 2, 'bus': 1}, t)
        self.controller.update_weighted_demand('Eastbound', {'truck': 3}, t)

        rl_state = self.controller.get_rl_state(self.int_name)
        self.assertIsNotNone(rl_state)
        self.assertIsInstance(rl_state, np.ndarray)

        num_lanes = 4
        num_vtypes = len(self.controller.vehicle_type_order)
        expected_len = (num_lanes * num_vtypes) + num_lanes + num_lanes + num_lanes
        self.assertEqual(len(rl_state), expected_len)


    def test_approach_counts_cleared_on_phase_completion(self):
        """Verify per-vehicle-type counts are cleared when a green phase completes."""
        t = 1.5
        self.controller.update_state(t)
        self.assertEqual(self.state['current_state'], 'GREEN')
        active_appr = self.state['config']['phases'][self.state['phases'][self.state['current_phase_index']]][0]
        
        # Inject vehicle counts on active approach
        self.controller.update_weighted_demand(active_appr, {'car': 4, 'truck': 2}, t)
        
        # Advance through GREEN -> YELLOW -> ALL_RED
        while self.state['current_state'] == 'GREEN':
            t += 1.0
            self.controller.update_state(t)
        while self.state['current_state'] == 'YELLOW':
            t += 1.0
            self.controller.update_state(t)
            
        self.assertEqual(self.state['current_state'], 'ALL_RED')
        # Counts for finished approach should now be 0
        stored = self.state['approach_counts_by_type'].get(active_appr, {})
        for vt, count in stored.items():
            self.assertEqual(count, 0)

    def test_wait_time_accumulation_on_weighted_demand(self):
        """Verify wait times accumulate on approaches with weighted demand."""
        t = 10.0
        self.controller.update_weighted_demand('Eastbound', {'car': 3}, t)
        # Advance 2 seconds while Northbound is green
        self.state['current_state'] = 'GREEN'
        self.state['current_phase_index'] = 0  # GreenForNorth
        
        self.state._update_wait_times(2.0, 'Northbound')
        self.assertAlmostEqual(self.state['approach_wait_time']['Eastbound'], 2.0)
        self.assertEqual(self.state['approach_wait_time']['Northbound'], 0.0)


    def test_round_robin_skips_manually_forced_red(self):
        """Verify fallback round-robin skips approaches that are manually forced red."""
        # Force Eastbound (phase index 1) to red
        self.controller.set_manual_override(self.int_name, 'Eastbound', is_forced_red=True)
        self.state['current_state'] = 'ALL_RED'
        self.state['current_phase_index'] = 0  # GreenForNorth
        
        # Advance to finish ALL_RED
        t = 2.0
        self.controller.update_state(t)
        self.assertEqual(self.state['current_state'], 'GREEN')
        # Should have skipped Eastbound and chosen Westbound (index 2)
        active_phase = self.state['phases'][self.state['current_phase_index']]
        self.assertEqual(active_phase, 'GreenForWest')

    def test_canonical_state_normalization(self):
        """Verify TrafficEnv.normalize_state_vector produces correct bounds and dimensions."""
        from rl_model.environment import TrafficEnv
        num_lanes = 4
        num_vtypes = 6
        raw_state = np.zeros((num_lanes * num_vtypes) + 3 * num_lanes, dtype=float)
        # 10 cars in lane 0
        raw_state[0] = 10.0
        # 150s wait time in lane 1
        raw_state[num_lanes * num_vtypes + 1] = 150.0
        # ambulance in lane 2
        raw_state[num_lanes * num_vtypes + num_lanes + 2] = 1.0
        # one-hot lane 3
        raw_state[-1] = 1.0

        norm = TrafficEnv.normalize_state_vector(raw_state, num_lanes, num_vtypes, normalize_max=50.0)
        self.assertEqual(len(norm), len(raw_state))
        self.assertAlmostEqual(norm[0], 0.2, places=3)
        self.assertTrue(0.0 < norm[num_lanes * num_vtypes + 1] < 1.0)
        self.assertEqual(norm[num_lanes * num_vtypes + num_lanes + 2], 1.0)
        self.assertEqual(norm[-1], 1.0)


if __name__ == '__main__':
    unittest.main()
