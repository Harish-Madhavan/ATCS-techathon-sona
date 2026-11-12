"""ATCS Core Traffic FSM & Controller Logic.

Provides:
- Global system configuration and timings (`config`)
- Intersection state machine & timing tracker (`IntersectionState`)
- Central traffic light controller & FSM coordinator (`TrafficLightController`)
- Structured application logging (`get_logger`)
"""
from core import config
from core.intersection_state import IntersectionState
from core.traffic_logic import TrafficLightController
from core.log import get_logger

__all__ = [
    "config",
    "IntersectionState",
    "TrafficLightController",
    "get_logger",
]
