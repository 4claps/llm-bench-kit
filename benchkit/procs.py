"""Child process helpers: every external command is launched from an argv
list (never a shell string), tracked by PID from launch, logged line by line
and killed with its whole process group on abort or timeout.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import Callable

# every live Proc, keyed by pid; kill_all() sweeps this after a run so no
# benchmark child can hold GPU memory into the next run
_registry: dict[int, "Proc"] = {}
_registry_lock = threading.Lock()


def _register(p: "Proc") -> None:
    with _registry_lock:
        _registry[p.pid] = p


def _unregister(p: "Proc") -> None:
    with _registry_lock:
        _registry.pop(p.pid, None)


def kill_all() -> list[str]:
    """Kill every still-running tracked process group (dead entries are
    dropped from the registry on the way). Returns a short description
    (pid + argv[0]) of each group that had to be killed, even when its
    leader already exited but children in the group still run."""
    with _registry_lock:
        tracked = list(_registry.values())
    killed = []
    for p in tracked:
        if p.group_alive():
            killed.append(f"{p.pid} {os.path.basename(str(p.argv[0]))}")
        p.kill()
    return killed


class Proc:
    def __init__(self, argv: list[str], env: dict | None = None, cwd: str | None = None,
                 on_line: Callable[[str], None] | None = None, preexec_uid: int | None = None,
                 split_stderr: bool = False):
        """split_stderr: keep stderr out of .lines (it still goes to on_line,
        prefixed "stderr: ") for callers that grade stdout, like the agent."""
        self.argv = argv
        kw: dict = {}
        if preexec_uid is not None and os.geteuid() == 0:
            kw["user"] = preexec_uid
            kw["group"] = preexec_uid
            kw["extra_groups"] = []
        self.p = subprocess.Popen(
            argv, env={**os.environ, **(env or {})}, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if split_stderr else subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            start_new_session=True, **kw)
        self.pid = self.p.pid
        self.lines: list[str] = []
        self._on_line = on_line
        self.err_lines: list[str] = []
        self._reader = threading.Thread(target=self._read, args=(self.p.stdout, self.lines, ""), daemon=True)
        self._reader.start()
        self._err_reader = None
        if split_stderr:
            self._err_reader = threading.Thread(target=self._read, args=(self.p.stderr, self.err_lines, "stderr: "),
                                                daemon=True)
            self._err_reader.start()
        _register(self)

    def _read(self, stream, store: list[str], prefix: str):
        for line in stream:
            line = line.rstrip("\n")
            store.append(line)
            if len(store) > 20000:
                del store[:10000]
            if self._on_line:
                try:
                    self._on_line(prefix + line)
                except Exception:  # noqa: BLE001
                    pass

    def alive(self) -> bool:
        return self.p.poll() is None

    def group_alive(self) -> bool:
        """Does the process group still have live members? The leader may
        have exited while its children keep running; a not-yet-reaped
        zombie leader must not count as alive."""
        self.p.poll()  # reap the leader if it already exited
        try:
            os.killpg(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def wait(self, timeout: float | None = None, abort: threading.Event | None = None) -> int | None:
        """Wait for exit; returns the exit code, or None on timeout/abort (the
        process group is killed in that case)."""
        t0 = time.time()
        while self.alive():
            if abort is not None and abort.is_set():
                self.kill()
                return None
            if timeout is not None and time.time() - t0 > timeout:
                self.kill()
                return None
            time.sleep(0.2)
        self._reader.join(timeout=5)
        if self._err_reader:
            self._err_reader.join(timeout=5)
        _unregister(self)
        return self.p.returncode

    def kill(self, grace: float = 10) -> None:
        """Kill the whole process group; never raises. Acts even when the
        group leader already exited: its children share the group and would
        otherwise survive it holding GPU memory. The grace wait and the
        SIGKILL step watch the group, not the leader, so a child that
        ignores SIGTERM still gets killed."""
        _unregister(self)
        self.p.poll()  # reap the leader if it already exited (see group_alive)
        try:
            os.killpg(self.pid, signal.SIGTERM)
        except OSError:  # group already gone
            pass
        t0 = time.time()
        while self.group_alive() and time.time() - t0 < grace:
            time.sleep(0.2)
        if self.group_alive():
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            self.p.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass

    def __enter__(self) -> "Proc":
        return self

    def __exit__(self, *exc) -> None:
        self.kill()
