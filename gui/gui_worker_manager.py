import multiprocessing as mp
import os
import numpy as np
from core import config
from core.log import get_logger
from rl_model.device import get_best_device_str
from vision.video_processor import process_video_worker

logger = get_logger(__name__)

def is_valid_polygon(polygon):
    try:
        arr = np.asarray(polygon)
        return arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] == 2
    except Exception:
        return False

class WorkerManager:
    def __init__(self, results_queue):
        self.results_queue = results_queue
        self.processes = []
        self.process_map = {}  # PID -> approach_name
        self.finished_workers = 0
        self.active_workers_initial_count = 0
        self.final_summaries = {}

    def start_workers(self, defined_polygons, on_worker_started_cb, on_worker_failed_cb):
        """
        Starts worker processes for defined approach polygons.
        defined_polygons: dict mapping approach_name -> polygon array
        on_worker_started_cb: function(approach_name, pid)
        on_worker_failed_cb: function(approach_name, exception)
        """
        self.active_workers_initial_count = 0
        self.processes.clear()
        self.process_map.clear()
        self.finished_workers = 0
        self.final_summaries.clear()

        device = get_best_device_str()
        logger.info("Starting %d workers (device=%s)...", len(defined_polygons), device)

        for approach_name, polygon in defined_polygons.items():
            video_path = next((path for name, path in config.VIDEO_PATHS if name == approach_name), None)
            if not video_path or not os.path.exists(video_path):
                err = RuntimeError(f"Video not found: {video_path}")
                logger.warning("Skipping %s: %s", approach_name, err)
                on_worker_failed_cb(approach_name, err)
                continue
            if not is_valid_polygon(polygon):
                err = ValueError(f"Invalid ROI polygon for {approach_name} (need Nx2, >=3 pts)")
                logger.warning("%s", err)
                on_worker_failed_cb(approach_name, err)
                continue
            
            p = mp.Process(
                target=process_video_worker,
                args=(
                    approach_name, video_path, config.MODEL_NAME, config.AMBULANCE_MODEL_NAME,
                    config.TARGET_CLASSES, config.AMBULANCE_CLASS_NAMES, config.CONFIDENCE_THRESHOLD,
                    config.PROCESS_EVERY_N_FRAMES, device, self.results_queue, polygon
                ),
                daemon=True
            )
            self.processes.append(p)
            try:
                p.start()
                self.process_map[p.pid] = approach_name
                self.active_workers_initial_count += 1
                on_worker_started_cb(approach_name, p.pid)
            except Exception as e:
                logger.exception("Failed to start %s: %s", approach_name, e)
                on_worker_failed_cb(approach_name, e)

        return self.active_workers_initial_count

    def check_dead_processes(self, on_dead_worker_cb):
        """
        Scans registered processes to identify and log unexpected terminations.
        on_dead_worker_cb: function(approach_name, pid)
        """
        if self.finished_workers < self.active_workers_initial_count:
            pids_to_check = list(self.process_map.keys())
            active_pids = {p.pid for p in self.processes if p.is_alive()}
            for pid in pids_to_check:
                if pid not in active_pids:
                    dead_approach_name = self.process_map[pid]
                    if dead_approach_name not in self.final_summaries:
                        logger.warning("Worker PID %s for %s terminated unexpectedly.", pid, dead_approach_name)
                        self.final_summaries[dead_approach_name] = {
                            'type': 'error',
                            'error': 'Process terminated unexpectedly',
                            'approach': dead_approach_name
                        }
                        self.finished_workers += 1
                        on_dead_worker_cb(dead_approach_name, pid)
                    del self.process_map[pid]

    def terminate_all(self):
        """
        Terminates all active worker processes on application close.
        """
        active_processes = [p for p in self.processes if p.is_alive()]
        if active_processes:
            logger.info("Terminating %d worker(s)...", len(active_processes))
            for p in active_processes:
                try:
                    p.terminate()
                    p.join(timeout=2.0)
                    if p.is_alive():
                        p.kill()
                        p.join(timeout=1.0)
                except Exception as e:
                    logger.warning("Error terminating PID %s: %s", getattr(p, "pid", "?"), e)
