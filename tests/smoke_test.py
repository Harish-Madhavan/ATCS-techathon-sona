import sys
import os
# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from gui import LaneCounterApp
import config
import numpy as np
from traffic_logic import TrafficLightController

# Simple in-process queue for testing
class SimpleQueue:
    def __init__(self):
        self._items = []
    def put(self, item):
        self._items.append(item)
    def get(self):
        return self._items.pop(0)
    def get_nowait(self):
        return self.get()
    def empty(self):
        return len(self._items) == 0

q = SimpleQueue()
root = tk.Tk()
app = LaneCounterApp(root, q)
# Create a simple TrafficLightController for testing
app.traffic_controller = TrafficLightController(config.TRAFFIC_LIGHT_CONFIG, config.VEHICLE_TYPE_WEIGHTS, config.DEFAULT_VEHICLE_WEIGHT)
# Create widgets without starting worker processes
app._create_widgets(root)
# Ensure polygons exist for approaches
for name,_ in config.VIDEO_PATHS:
    app.defined_polygons[name] = np.array([[0,0],[1,0],[1,1],[0,1]])

keys = list(app.approach_widgets.keys())
print('approach widget keys:', keys)
if keys:
    first = keys[0]
    q.put({'type':'status_update','approach':first,'status':'Models Loaded'})
    q.put({'type':'lane_update','approach':first,'in_lane_current_frame_agg':2,'counts_by_type':{'car':2},'ambulance_detected':False})
    q.put({'type':'final_summary','approach':first,'total_vehicles_in_lane_agg':10,'total_counts_by_type':{'car':8,'bus':2}})
    app.update_traffic_logic()
    print('Status for', first, app.approach_widgets[first]['status'].cget('text'))
else:
    print('No approach widgets found')

root.destroy()
