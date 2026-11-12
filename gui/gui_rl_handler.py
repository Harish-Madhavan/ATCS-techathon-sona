"""Deprecated shim — RL inference now lives in rl_model.inference.

Kept so existing imports keep working. New code should use:
    from rl_model.inference import RLDecisionMaker
"""
from rl_model.inference import RLDecisionMaker

__all__ = ["RLDecisionMaker"]
