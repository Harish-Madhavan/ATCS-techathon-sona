"""ATCS Graphical Operator Interface & Dashboard.

Provides:
- Main Tkinter dashboard application (`LaneCounterApp`)
- Startup and ROI calibration dialog (`SetupDialog`)
- Live traffic light and trend plotting widgets (`TrafficLightWidget`, `TrendPlotWidget`)
- Dark theme application helper (`apply_theme`)
"""
from gui.gui import LaneCounterApp
from gui.gui_setup import SetupDialog
from gui.gui_components import TrafficLightWidget, TrendPlotWidget
from gui.theme import apply_theme

__all__ = [
    "LaneCounterApp",
    "SetupDialog",
    "TrafficLightWidget",
    "TrendPlotWidget",
    "apply_theme",
]
