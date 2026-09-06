"""Nonblocking host-local admission control for complete dashboard runs.

F1SIM_MAX_CONCURRENT_RUNS is an integer from 1 to 4 (default 1).
F1SIM_RUN_LOCK_DIR defaults to ``tempfile.gettempdir()/f1sim-run-capacity``.
All service processes must use the same limit and directory on a local filesystem,
with read/write access. Containers must share that directory to share capacity;
this is not a distributed limiter. Configuration is read when the app is built.

Empty slot files contain no live data and must not be deleted while servers run.
OS advisory locks coordinate threads and processes. Normal release explicitly
unlocks, even if a forked worker inherited the descriptor. After an abrupt POSIX
process exit, inherited descriptors can retain its lock until workers also exit.
The limit counts API requests, including their live I/O and all scenarios, not
individual Monte Carlo child workers. CLI/direct Python simulations are separate.
"""

from __future__ import annotations

import errno
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class RunCapacity:
    """Hold one independent file descriptor and OS lock per admitted request."""

    def __init__(self, limit: int, directory: Path) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 4:
            raise ValueError("F1SIM_MAX_CONCURRENT_RUNS must be an integer between 1 and 4")
        self.limit = limit
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_environment(cls) -> RunCapacity:
        """Fail at app construction on invalid capacity configuration."""
        try:
            limit = int(os.environ.get("F1SIM_MAX_CONCURRENT_RUNS", "1"))
        except ValueError as exc:
            raise ValueError(
                "F1SIM_MAX_CONCURRENT_RUNS must be an integer between 1 and 4"
            ) from exc
        directory = Path(os.environ.get(
            "F1SIM_RUN_LOCK_DIR", str(Path(tempfile.gettempdir()) / "f1sim-run-capacity")
        ))
        return cls(limit, directory)

    @contextmanager
    def acquire(self) -> Iterator[bool]:
        """Yield immediately with availability; always close an acquired slot."""
        held_fd: int | None = None
        try:
            for slot in range(self.limit):
                fd = os.open(self.directory / f"slot-{slot}.lock", os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    if os.name == "nt":
                        # Windows supports locking a range beyond EOF, keeping files empty.
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    else:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    os.close(fd)
                    if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        continue
                    raise
                held_fd = fd
                break
            yield held_fd is not None
        finally:
            if held_fd is not None:
                try:
                    if os.name == "nt":
                        msvcrt.locking(held_fd, msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(held_fd, fcntl.LOCK_UN)
                finally:
                    os.close(held_fd)
