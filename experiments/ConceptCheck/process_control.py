"""Clean up children started by this controller, including Windows launchers."""

import os
import subprocess


def terminate_process_tree(process, timeout=15):
    """Stop an owned Popen child and report the achieved cleanup scope.

    Windows venv Python may expose a launcher PID rather than the interpreter
    PID. Try taskkill /T while the owned launcher is alive. If OS permissions
    prevent this, use Popen's existing process handle and report direct-child
    cleanup only: descendants are not guaranteed to stop in this fallback.
    Other platforms also use ordinary terminate/wait/kill for the direct child.
    A child that still cannot be stopped raises instead of hiding the failure.
    """
    if timeout <= 0:
        raise ValueError("cleanup timeout must be positive")
    if process.poll() is not None:
        return {"cleanup_scope": "direct-child", "cleanup_method": "already-exited"}
    tree_error = None
    if os.name == "nt":
        taskkill = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                                "System32", "taskkill.exe")
        try:
            result = subprocess.run(
                [taskkill, "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, text=True, errors="replace", timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            tree_error = str(error)
        else:
            if result.returncode:
                tree_error = result.stderr.strip() or result.stdout.strip() or f"taskkill exit {result.returncode}"
            else:
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired as error:
                    tree_error = str(error)
                else:
                    return {"cleanup_scope": "tree", "cleanup_method": "taskkill"}
    method = "already-exited"
    if process.poll() is None:
        process.terminate()
        method = "terminate"
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)
            method = "kill"
    result = {"cleanup_scope": "direct-child", "cleanup_method": method}
    if tree_error is not None:
        result["tree_cleanup_error"] = tree_error
    return result
