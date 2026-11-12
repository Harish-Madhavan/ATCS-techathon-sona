import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.traffic_logic import TrafficLightController
from core import config
import time

ctrl = TrafficLightController(config.TRAFFIC_LIGHT_CONFIG, config.VEHICLE_TYPE_WEIGHTS, config.DEFAULT_VEHICLE_WEIGHT)
state = ctrl.intersections['Intersection1']

# Helper to force ALL_RED and call update_state with action
def test_action(action, label):
    state['current_state'] = 'ALL_RED'
    state['state_timer'] = state['config']['timings']['all_red'] + 0.1
    print('\n--- TEST', label, 'action=', action)
    ctrl.update_state(time.time(), rl_action=action)
    print('Resulting current_cycle_max_green =', state['current_cycle_max_green'])
    print('current_phase_index =', state['current_phase_index'])

duration_options = config.DURATION_OPTIONS
num_durations = len(duration_options)
# action: lane 0, duration index 0 (10s)
action_10s = 0 * num_durations + 0
# action: lane 0, duration index 2 (30s)
action_30s = 0 * num_durations + 2

print('Duration options:', duration_options)
print('Initial current_cycle_max_green =', state['current_cycle_max_green'])

test_action(action_10s, '10s')
test_action(action_30s, '30s')
