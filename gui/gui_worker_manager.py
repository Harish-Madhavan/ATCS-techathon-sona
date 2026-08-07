import multiprocessing as mp
import os
import time
import traceback
import torch
from vision.video_processor import process_video_worker
from core import config

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

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[WorkerManager] Starting parallel processing for {len(defined_polygons)} approaches using device hint '{device}'...")

        for approach_name, polygon in defined_polygons.items():
            video_path = next((path for name, path in config.VIDEO_PATHS if name == approach_name), None)
            if not video_path:
                print(f"[WorkerManager Error] Missing video path for {approach_name}. Skipping.")
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
                print(f"[WorkerManager Error] Failed to start process for {approach_name}: {e}")
                traceback.print_exc()
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
                        print(f"\n!!! [WorkerManager Error] Worker PID {pid} for {dead_approach_name} terminated unexpectedly. !!!")
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
            print(f"[WorkerManager] Terminating {len(active_processes)} active worker process(es)...")
            for p in active_processes:
                try:
                    p.terminate()
                    p.join(timeout=1.0)
                except Exception as e:
                    print(f"[WorkerManager] Error terminating PID {p.pid if p else '?'}: {e}")
