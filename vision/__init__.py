"""ATCS Computer Vision & Perception Pipeline.

Provides:
- Multiprocessed YOLO inference and lane occupancy worker (`process_video_worker`)
- Interactive point-in-polygon ROI configuration tool (`define_polygon_interactive`)
"""
from vision.video_processor import process_video_worker
from vision.polygon_utils import define_polygon_interactive

__all__ = [
    "process_video_worker",
    "define_polygon_interactive",
]
