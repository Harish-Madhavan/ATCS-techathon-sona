"""Main entrypoint for the Adaptive Traffic Control System (ATCS)."""
import argparse
import multiprocessing as mp
import os
import sys
import tkinter as tk
import cv2

from core import config
from core.log import get_logger
from gui.gui import LaneCounterApp

logger = get_logger(__name__)


def setup_windows_dpi():
    """Enable DPI awareness on Windows to prevent blurry rendering."""
    if os.name == 'nt':
        try:
            from ctypes import windll
            PROCESS_SYSTEM_DPI_AWARE = 1
            windll.shcore.SetProcessDpiAwareness(PROCESS_SYSTEM_DPI_AWARE)
            logger.debug("DPI awareness set (shcore).")
        except (ImportError, AttributeError, OSError):
            try:
                windll.user32.SetProcessDPIAware()
                logger.debug("DPI awareness set (user32).")
            except Exception as e2:
                logger.debug("DPI awareness unavailable: %s", e2)


def setup_multiprocessing():
    """Ensure safe multiprocessing start method across operating systems."""
    try:
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None:
            mp.set_start_method('spawn', force=True)
            logger.debug("Multiprocessing set to 'spawn'.")
        elif current_method != 'spawn':
            try:
                mp.set_start_method('spawn', force=True)
                logger.debug("Multiprocessing forced to 'spawn'.")
            except RuntimeError as re:
                logger.warning("Could not force 'spawn' (%s); using '%s'.", re, current_method)
    except Exception as e:
        logger.warning("Multiprocessing start-method issue: %s", e)


def parse_args(argv=None):
    """Parse command line arguments for ATCS."""
    parser = argparse.ArgumentParser(description="Adaptive Traffic Light Control System")
    parser.add_argument("--headless", action="store_true", help="Run application in headless mode (hides GUI window)")
    parser.add_argument("--no-rl", action="store_true", help="Disable Reinforcement Learning model control")
    parser.add_argument("--esp32", type=str, help="Enable ESP32 serial communication on the specified COM port (e.g. COM3)")
    parser.add_argument("--video", type=str, help="Specify a single video path to use for all approaches")
    return parser.parse_args(argv)


def main(argv=None):
    """Run the ATCS application."""
    args = parse_args(argv)

    if args.headless:
        os.environ["LANE_COUNTER_HEADLESS"] = "1"
        logger.info("Headless mode enabled.")
    if args.no_rl:
        config.RL_MODEL_ENABLED = False
        logger.info("RL control disabled.")
    if args.esp32:
        config.ESP32_ENABLED = True
        config.ESP32_PORT = args.esp32
        logger.info("ESP32 enabled on port '%s'.", args.esp32)
    if args.video:
        if os.path.exists(args.video):
            config.VIDEO_PATHS = [
                ("Northbound", args.video),
                ("Eastbound", args.video),
                ("Westbound", args.video),
                ("Southbound", args.video),
            ]
            logger.info("Video override: '%s'.", args.video)
        else:
            logger.error("Video file does not exist: %s", args.video)
            return 1

    setup_multiprocessing()
    setup_windows_dpi()

    manager = mp.Manager()
    results_queue = manager.Queue()
    root = tk.Tk()
    app = LaneCounterApp(root, results_queue)

    if app.initialize_application():
        logger.info("Initialization complete. Starting main loop...")
        root.mainloop()
        logger.info("Main loop exited.")

    cv2.destroyAllWindows()
    logger.info("Script finished.")
    return 0


if __name__ == '__main__':
    sys.exit(main())