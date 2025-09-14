import tkinter as tk
from tkinter import ttk, font, messagebox, scrolledtext
import multiprocessing as mp
from queue import Empty
import time
import os
import traceback
from collections import defaultdict, deque 
import datetime as dt 
import torch 
from functools import partial 

try:
    import matplotlib
    matplotlib.use('TkAgg') 
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
    print("[GUI] Matplotlib found and loaded.")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("\nWARNING: Matplotlib not found. Install it ('pip install matplotlib') to enable plots.\n")

import config 
from video_processor import process_video_worker 
from traffic_logic import TrafficLightController
from polygon_utils import define_polygon_interactive
from rl_model.train import QNetwork

if config.ESP32_ENABLED:
    try:
        from esp32_controller import ESP32SerialController
        ESP32_CONTROLLER_AVAILABLE = True
        print("[GUI] ESP32SerialController found and will be used.")
    except ImportError:
        ESP32_CONTROLLER_AVAILABLE = False
        print("\nWARNING: esp32_controller.py not found or has issues. ESP32 communication will be disabled.\n")
    except Exception as e_esp_import: 
        ESP32_CONTROLLER_AVAILABLE = False
        print(f"\nWARNING: Error importing esp32_controller.py: {e_esp_import}. ESP32 communication will be disabled.\n")
else:
    ESP32_CONTROLLER_AVAILABLE = False
    print("[GUI] ESP32 communication is disabled in config.py.")

class LaneCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Adaptive Traffic Light Control - {config.VERSION}")
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=config.DEFAULT_FONT_SIZE)
        self.root.option_add("*Font", default_font)

        self.center_window(config.INITIAL_WINDOW_WIDTH, config.INITIAL_WINDOW_HEIGHT)

        self.style = ttk.Style()
        try:
            default_trough = self.style.lookup("Horizontal.TProgressbar", "troughcolor")
            sys_default_bar_bg = self.style.lookup("Horizontal.TProgressbar", "background")
            default_bar_bg = "blue" if sys_default_bar_bg in ['green', 'forest green', '#008000', '#228B22'] else sys_default_bar_bg
            if default_bar_bg is None: default_bar_bg = "blue"
        except tk.TclError:
            default_trough = "lightgrey"
            default_bar_bg = "blue"
            print("[GUI Style] TclError looking up Progressbar style, using hardcoded fallbacks.")
        
        self.style.configure("green.Horizontal.TProgressbar", background='forest green', troughcolor=default_trough)
        self.style.configure("yellow.Horizontal.TProgressbar", background='darkorange', troughcolor=default_trough)
        self.style.configure("red.Horizontal.TProgressbar", background='red', troughcolor=default_trough)
        self.style.configure("default.Horizontal.TProgressbar", background=default_bar_bg, troughcolor=default_trough)

        self.defined_polygons = {}
        self.skipped_approaches = []
        self.processes = []
        self.process_map = {}
        self.final_summaries = {}
        self.approach_widgets = {}
        self.traffic_light_ui = {}
        self.finished_workers = 0
        self.active_workers_initial_count = 0
        self.traffic_logic_timer_id = None
        self.plot_update_timer_id = None
        self.esp32_controller = None
        self.manual_overrides_gui_state = defaultdict(bool) 

        self.approach_history = defaultdict(lambda: deque(maxlen=config.PLOT_MAX_POINTS))
        self.manager = mp.Manager()
        self.results_queue = self.manager.Queue()

        if config.ESP32_ENABLED and ESP32_CONTROLLER_AVAILABLE:
            configured_video_approaches = [name for name, path in config.VIDEO_PATHS]
            valid_esp32_mapping = {}
            for gui_approach, esp_code in config.ESP32_APPROACH_MAPPING.items():
                if gui_approach in configured_video_approaches:
                    valid_esp32_mapping[gui_approach] = esp_code
                else:
                    print(f"[GUI Warning] ESP32_APPROACH_MAPPING contains key '{gui_approach}' which is not in VIDEO_PATHS. It will be ignored for ESP32.")
            
            if not valid_esp32_mapping and config.ESP32_APPROACH_MAPPING:
                messagebox.showwarning("ESP32 Config Warning",
                                       "ESP32_APPROACH_MAPPING in config.py does not contain any valid approaches defined in VIDEO_PATHS. ESP32 will not be used.")

    def center_window(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)
        self.root.geometry(f'{width}x{height}+{int(x)}+{int(y)}')

    def initialize_application(self):
        """
        Initializes the main application components, including UI, traffic logic,
        and video processing workers.
        """
        # Create main frames
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Initialize TrafficLightController
        self.traffic_controller = TrafficLightController(
            config.TRAFFIC_LIGHT_CONFIG,
            config.VEHICLE_TYPE_WEIGHTS,
            config.DEFAULT_VEHICLE_WEIGHT
        )

        # Define polygons for each video feed
        for approach_name, video_path in config.VIDEO_PATHS:
            polygon = define_polygon_interactive(video_path, approach_name)
            if polygon:
                self.defined_polygons[approach_name] = polygon
            else:
                self.skipped_approaches.append(approach_name)
                messagebox.showwarning("Polygon Definition Skipped", f"No polygon defined for {approach_name}. This approach will be skipped.")

        # Start video processing workers
        for approach_name, video_path in config.VIDEO_PATHS:
            if approach_name in self.skipped_approaches:
                continue
            
            p = mp.Process(target=process_video_worker, args=(
                video_path,
                self.results_queue,
                self.defined_polygons[approach_name],
                approach_name,
                config.MODEL_NAME,
                config.TARGET_CLASSES,
                config.CONFIDENCE_THRESHOLD,
                config.PROCESS_EVERY_N_FRAMES
            ))
            self.processes.append(p)
            self.process_map[approach_name] = p

        if not self.processes:
            messagebox.showerror("Initialization Error", "No approaches to monitor. Exiting.")
            return False

        for p in self.processes:
            p.start()

        self.active_workers_initial_count = len(self.processes)
        
        # Create UI components
        self._create_widgets(main_frame)

        # Start periodic updates
        self.traffic_logic_timer_id = self.root.after(config.TRAFFIC_LOGIC_UPDATE_INTERVAL_MS, self.update_traffic_logic)
        if MATPLOTLIB_AVAILABLE and config.PLOT_ENABLE:
            self.plot_update_timer_id = self.root.after(config.PLOT_UPDATE_INTERVAL_MS, self.update_plots)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        return True

    def _create_widgets(self, parent_frame):
        """Creates the UI widgets for the application."""
        # This is a placeholder for the UI creation.
        # Based on the __init__ method, it should create frames for:
        # - Approach monitors (approach_widgets)
        # - Traffic light status (traffic_light_ui)
        # - Plots (if matplotlib is available)
        # - A log window
        
        # Example for one approach
        for approach_name, _ in config.VIDEO_PATHS:
            if approach_name in self.skipped_approaches:
                continue
            
            frame = ttk.LabelFrame(parent_frame, text=approach_name)
            frame.pack(padx=5, pady=5, fill=tk.X)
            
            label = ttk.Label(frame, text="Waiting for data...")
            label.pack()
            
            self.approach_widgets[approach_name] = {'label': label}

    def update_traffic_logic(self):
        """Periodically updates the traffic logic."""
        # This is a placeholder for the traffic logic update.
        # It should:
        # - Get data from the results_queue
        # - Update the traffic_controller
        # - Get the RL action if RL_MODEL_ENABLED is True
        # - Call traffic_controller.update_state()
        # - Update the UI
        
        # Get RL action
        rl_action = None
        if config.RL_MODEL_ENABLED:
            # This is a placeholder for getting the RL action
            # In a real implementation, you would get the state from the traffic_controller,
            # and use the RL model to choose an action.
            pass

        # Update traffic controller
        self.traffic_controller.update_state(time.time(), rl_action)
        
        # Reschedule the update
        self.traffic_logic_timer_id = self.root.after(config.TRAFFIC_LOGIC_UPDATE_INTERVAL_MS, self.update_traffic_logic)

    def update_plots(self):
        """Periodically updates the plots."""
        # This is a placeholder for plot updates.
        
        # Reschedule the update
        if MATPLOTLIB_AVAILABLE and config.PLOT_ENABLE:
            self.plot_update_timer_id = self.root.after(config.PLOT_UPDATE_INTERVAL_MS, self.update_plots)

    def on_closing(self):
        """Handles the window closing event."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            # Stop periodic updates
            if self.traffic_logic_timer_id:
                self.root.after_cancel(self.traffic_logic_timer_id)
            if self.plot_update_timer_id:
                self.root.after_cancel(self.plot_update_timer_id)

            # Terminate worker processes
            for p in self.processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1) # Wait for termination

            self.root.destroy()