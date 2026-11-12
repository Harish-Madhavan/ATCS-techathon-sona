import unittest
import numpy as np


class TestWorkerPolygonValidation(unittest.TestCase):
    def test_valid_and_invalid(self):
        from gui.gui_worker_manager import is_valid_polygon
        self.assertTrue(is_valid_polygon(np.array([[0, 0], [10, 0], [10, 10]])))
        self.assertFalse(is_valid_polygon(np.array([[0, 0], [1, 1]])))
        self.assertFalse(is_valid_polygon(np.array([1, 2, 3])))
        self.assertFalse(is_valid_polygon(None))

    def test_start_workers_rejects_bad_polygon_without_spawn(self):
        import queue
        from gui.gui_worker_manager import WorkerManager
        q = queue.Queue()
        wm = WorkerManager(q)
        calls = []
        n = wm.start_workers(
            {"Northbound": np.array([[0, 0], [1, 1]])},
            on_worker_started_cb=lambda a, p: calls.append(("started", a)),
            on_worker_failed_cb=lambda a, e: calls.append(("failed", a)),
        )
        self.assertEqual(n, 0)
        self.assertEqual(calls, [("failed", "Northbound")])


class TestSetupBrowseDir(unittest.TestCase):
    def test_falls_back_to_assets_or_root(self):
        import os
        from gui.gui_setup import _default_browse_dir
        d = _default_browse_dir("/nonexistent/path.mp4")
        self.assertTrue(os.path.isdir(d))
        # Existing file's directory wins
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            want = os.path.dirname(f.name)
        try:
            self.assertEqual(_default_browse_dir(f.name), want)
        finally:
            os.unlink(f.name)


class TestContractHelpersForUI(unittest.TestCase):
    def test_esp32_command_stable(self):
        from hardware.esp32_controller import ESP32SerialController
        c = ESP32SerialController(port="MOCK", baudrate=115200,
                                  approach_mapping={"B": "B", "A": "A"})
        # Insertion order shouldn't matter; output sorted.
        cmd1 = c.build_command({"B": {"state": "GREEN"}, "A": {"state": "RED"}})
        cmd2 = c.build_command({"A": {"state": "RED"}, "B": {"state": "GREEN"}})
        self.assertEqual(cmd1, cmd2)
        self.assertEqual(cmd1, "A:R,B:G\n")
        c.close()


if __name__ == "__main__":
    unittest.main()
