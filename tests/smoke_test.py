import sys
import os
# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from gui.gui import LaneCounterApp
from core import config
import numpy as np
from core.traffic_logic import TrafficLightController

from queue import Empty

# Simple in-process queue for testing
class SimpleQueue:
    def __init__(self):
        self._items = []
    def put(self, item):
        self._items.append(item)
    def get(self):
        if not self._items:
            raise Empty()
        return self._items.pop(0)
    def get_nowait(self):
        return self.get()
    def empty(self):
        return len(self._items) == 0

q = SimpleQueue()
root = tk.Tk()
app = LaneCounterApp(root, q)
# Create a simple TrafficLightController for testing
app.controller = TrafficLightController(config.TRAFFIC_LIGHT_CONFIG, config.VEHICLE_TYPE_WEIGHTS, config.DEFAULT_VEHICLE_WEIGHT)
# Ensure polygons exist for approaches before creating widgets
for name,_ in config.VIDEO_PATHS:
    app.defined_polygons[name] = np.array([[0,0],[1,0],[1,1],[0,1]])

# Create widgets without starting worker processes
app._create_widgets()

keys = list(app.approach_widgets.keys())
print('approach widget keys:', keys)
if keys:
    first = keys[0]
    q.put({'type':'status_update','approach':first,'status':'Models Loaded'})
    q.put({'type':'lane_update','approach':first,'in_lane_current_frame_agg':2,'counts_by_type':{'car':2},'ambulance_detected':False})
    q.put({'type':'final_summary','approach':first,'total_vehicles_in_lane_agg':10,'total_counts_by_type':{'car':8,'bus':2}})
    
    # Process the mock queue messages
    app._check_queue()
    # Update displays
    app._update_traffic_light_display()
    
    # In newer tkinter/python versions, widget text retrieval uses winfo/cget or StringVars
    status_text = app.approach_widgets[first]['vars']['status'].get()
    print('Status for', first, status_text)
else:
    print('No approach widgets found')

root.destroy()
