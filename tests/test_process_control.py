"""Owned child cleanup, including a real Windows venv process tree."""

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from experiments.ConceptCheck import process_control


class ProcessControlTests(unittest.TestCase):
    def test_completed_child_requires_no_cleanup(self):
        child = Mock(pid=4321)
        child.poll.return_value = 0
        with patch.object(process_control.subprocess, "run") as run:
            result = process_control.terminate_process_tree(child)
        self.assertEqual(result["cleanup_scope"], "direct-child")
        run.assert_not_called()
        child.terminate.assert_not_called()

    def test_windows_cleanup_targets_only_the_owned_tree(self):
        child = Mock(pid=4321)
        child.poll.return_value = None
        with patch.object(process_control.os, "name", "nt"), patch.object(
                process_control.subprocess, "run", return_value=Mock(returncode=0)) as run:
            result = process_control.terminate_process_tree(child, timeout=3)
        self.assertEqual(result, {"cleanup_scope": "tree", "cleanup_method": "taskkill"})
        command = run.call_args.args[0]
        self.assertEqual(command[1:], ["/PID", "4321", "/T", "/F"])
        self.assertTrue(command[0].lower().endswith("taskkill.exe"))
        self.assertEqual(run.call_args.kwargs["timeout"], 3)
        child.wait.assert_called_once_with(timeout=3)
        child.terminate.assert_not_called()

    def test_access_denied_uses_owned_handle_and_reports_limited_scope(self):
        child = Mock(pid=4321)
        child.poll.return_value = None
        result = Mock(returncode=1, stderr="access denied", stdout="")
        with patch.object(process_control.os, "name", "nt"), patch.object(
                process_control.subprocess, "run", return_value=result):
            cleanup = process_control.terminate_process_tree(child)
        self.assertEqual(cleanup, {"cleanup_scope": "direct-child", "cleanup_method": "terminate",
                                   "tree_cleanup_error": "access denied"})
        child.wait.assert_called_once_with(timeout=15)
        child.terminate.assert_called_once_with()

    def test_child_exiting_during_windows_cleanup_is_not_an_error(self):
        child = Mock(pid=4321)
        child.poll.side_effect = [None, 0]
        with patch.object(process_control.os, "name", "nt"), patch.object(
                process_control.subprocess, "run", return_value=Mock(returncode=128, stderr="gone", stdout="")):
            cleanup = process_control.terminate_process_tree(child)
        self.assertEqual(cleanup["cleanup_scope"], "direct-child")
        self.assertEqual(cleanup["cleanup_method"], "already-exited")
        child.terminate.assert_not_called()

    def test_posix_cleanup_kills_after_grace_period(self):
        child = Mock(pid=4321)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("child", 2), 0]
        with patch.object(process_control.os, "name", "posix"):
            cleanup = process_control.terminate_process_tree(child, timeout=2)
        self.assertEqual(cleanup, {"cleanup_scope": "direct-child", "cleanup_method": "kill"})
        child.terminate.assert_called_once_with()
        child.kill.assert_called_once_with()
        self.assertEqual(child.wait.call_count, 2)

    @unittest.skipUnless(os.name == "nt", "Windows venv launcher process-tree check")
    def test_real_venv_cleanup_reports_scope_and_reaps_the_owned_launcher(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)

        def alive(pid):
            handle = kernel.OpenProcess(0x1000, False, pid)
            if not handle:
                if ctypes.get_last_error() == 87:
                    return False
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                status = wintypes.DWORD()
                if not kernel.GetExitCodeProcess(handle, ctypes.byref(status)):
                    raise ctypes.WinError(ctypes.get_last_error())
                return status.value == 259
            finally:
                kernel.CloseHandle(handle)

        with tempfile.TemporaryDirectory() as directory:
            root_marker = Path(directory) / "root.json"
            grandchild_marker = Path(directory) / "grandchild.json"
            grandchild_code = (
                "import json,os,sys,time; from pathlib import Path; "
                "Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid()})); time.sleep(12)"
            )
            root_code = (
                "import json,os,subprocess,sys,time; from pathlib import Path; "
                "child=subprocess.Popen([sys.executable,'-c',sys.argv[3],sys.argv[2]]); "
                "Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid()})); time.sleep(12)"
            )
            child = subprocess.Popen([sys.executable, "-c", root_code, str(root_marker),
                                      str(grandchild_marker), grandchild_code])
            pids = []
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        pids = [json.loads(path.read_text())["pid"]
                                for path in (root_marker, grandchild_marker)]
                        break
                    except (OSError, ValueError):
                        time.sleep(0.05)
                self.assertEqual(len(pids), 2, "both bounded CPU helpers must have started")
                self.assertTrue(all(alive(pid) for pid in pids))
                cleanup = process_control.terminate_process_tree(child, timeout=5)
                self.assertIn(cleanup["cleanup_scope"], ("tree", "direct-child"))
                self.assertIsNotNone(child.poll(), "the owned Popen child must have exited")
                deadline = time.monotonic() + 2
                while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
                    time.sleep(0.05)
                if cleanup["cleanup_scope"] == "tree":
                    self.assertFalse(any(alive(pid) for pid in pids), "a descendant survived tree cleanup")
                else:
                    # Access-denied fallback intentionally promises only the
                    # owned launcher's exit; a nested sleeper can remain alive.
                    self.assertIn("tree_cleanup_error", cleanup)
            finally:
                if child.poll() is None:
                    process_control.terminate_process_tree(child, timeout=5)
                # Helpers exit by themselves even if a regression leaves an orphan.
                deadline = time.monotonic() + 14
                while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
                    time.sleep(0.1)


if __name__ == "__main__":
    unittest.main()
