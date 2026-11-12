import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
from core.traffic_logic import TrafficLightController
from core import config


def run_sim(duration=120):
    ctrl = TrafficLightController(config.TRAFFIC_LIGHT_CONFIG)
    int_names = ctrl.get_intersection_names()
    print('Intersections:', int_names)
    intersection = int_names[0]

    # simulate starting at t=0
    t = 0.0
    step = 1.0

    # Let it run initial ALL_RED for all_red seconds
    all_red = ctrl.intersections[intersection]['config']['timings']['all_red']
    print('all_red', all_red)

    # Determine rl action encoding ahead of time so we can pass it at the ALL_RED -> GREEN
    duration_options = getattr(config, 'DURATION_OPTIONS', [10, 20, 30])
    num_dur = len(duration_options)
    chosen_lane = 0
    chosen_duration_idx = duration_options.index(30)
    rl_action = chosen_lane * num_dur + chosen_duration_idx

    # First, run until ALL_RED finishes and pass the RL action exactly when ALL_RED ends
    while t < all_red + 0.1:
        # pass RL action at the moment ALL_RED finishes (t >= all_red)
        if t >= all_red:
            ctrl.update_state(t, rl_action=rl_action)
        else:
            ctrl.update_state(t, rl_action=None)
        print(t, ctrl.get_intersection_status(intersection))
        t += step

    print('Applying RL action', rl_action, '-> duration', duration_options[chosen_duration_idx])

    # Advance for several seconds and print status
    for i in range(0, 60):
        ctrl.update_state(t, rl_action=None)
        status = ctrl.get_intersection_status(intersection)
        print(f"t={t:4.0f}s status={status['state']} phase={status['phase']} timer={status['timer']:.1f}/{status['max_duration']}")
        t += step
        time.sleep(0.01)


if __name__ == '__main__':
    run_sim(90)
