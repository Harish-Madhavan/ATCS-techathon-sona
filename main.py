import tkinter as tk
import multiprocessing as mp
import os
import sys
import cv2 
import argparse
from core import config
from gui.gui import LaneCounterApp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adaptive Traffic Light Control System")
    parser.add_argument("--headless", action="store_true", help="Run application in headless mode (hides GUI window)")
    parser.add_argument("--no-rl", action="store_true", help="Disable Reinforcement Learning model control")
    parser.add_argument("--esp32", type=str, help="Enable ESP32 serial communication on the specified COM port (e.g. COM3)")
    parser.add_argument("--video", type=str, help="Specify a single video path to use for all approaches")
    
    args = parser.parse_args()
    
    if args.headless:
        os.environ["LANE_COUNTER_HEADLESS"] = "1"
        print("[Main] Headless mode enabled via command line.")
    if args.no_rl:
        config.RL_MODEL_ENABLED = False
        print("[Main] RL Model control disabled via command line.")
    if args.esp32:
        config.ESP32_ENABLED = True
        config.ESP32_PORT = args.esp32
        print(f"[Main] ESP32 communication enabled on port '{args.esp32}' via command line.")
    if args.video:
        if os.path.exists(args.video):
            config.VIDEO_PATHS = [
                ("Northbound", args.video),
                ("Eastbound", args.video),
                ("Westbound", args.video),
                ("Southbound", args.video),
            ]
            print(f"[Main] Overriding video paths for all approaches to: '{args.video}'")
        else:
            print(f"[Main Error] Specified video file does not exist: {args.video}")
            sys.exit(1)

    try:
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None:
             mp.set_start_method('spawn', force=True)
             print("[Main] Set multiprocessing start method to 'spawn'.")
        elif current_method != 'spawn':
             
             try:
                 mp.set_start_method('spawn', force=True)
                 print("[Main] Forcing multiprocessing start method to 'spawn'.")
             except RuntimeError as re:
                 print(f"[Main] Warning: Could not force 'spawn' ({re}). Using existing '{current_method}'.")
        else:
            print(f"[Main] Multiprocessing start method already set to 'spawn'.")
    except Exception as e: 
        print(f"[Main] Warning: Issue setting start method ('{mp.get_start_method(allow_none=True)}'): {e}.")


    if os.name == 'nt':
        try:
            from ctypes import windll
            
            PROCESS_SYSTEM_DPI_AWARE = 1
            windll.shcore.SetProcessDpiAwareness(PROCESS_SYSTEM_DPI_AWARE)
            print("[Main] Set DPI Awareness (shcore System Aware)")
        except (ImportError, AttributeError, OSError):
            try:
                
                windll.user32.SetProcessDPIAware()
                print("[Main] Set DPI Awareness (user32)")
            except Exception as e2:
                 print(f"[Main] Could not set DPI awareness: {e2}")

    
    manager = mp.Manager()
    results_queue = manager.Queue()
    root = tk.Tk()
    app = LaneCounterApp(root, results_queue) 

    
    
    if app.initialize_application(): 
         print("[Main] Initialization complete. Starting main loop...")
         root.mainloop() 
         print("[Main] Main loop exited.") 
    cv2.destroyAllWindows()
    print("[Main] Script finished.")