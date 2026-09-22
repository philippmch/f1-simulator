"""Low-level cooperative cancellation primitives for simulation work."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator

CancellationCallback = Callable[[], bool]
_POLL_INTERVAL = 16


class SimulationCancelled(RuntimeError):
    """Raised when a simulation is cancelled before a complete result exists."""


@dataclass
class _PollState:
    remaining: int = 0


_callback: ContextVar[CancellationCallback | None] = ContextVar(
    "f1sim_cancellation_callback", default=None,
)
_poll_state: ContextVar[_PollState | None] = ContextVar(
    "f1sim_cancellation_poll_state", default=None,
)


@contextmanager
def cancellation_scope(callback: CancellationCallback | None) -> Iterator[None]:
    """Install a cancellation callback for this context and restore it on exit."""
    callback_token = _callback.set(callback)
    poll_token = _poll_state.set(_PollState())
    try:
        yield
    finally:
        _poll_state.reset(poll_token)
        _callback.reset(callback_token)


def install_cancellation_callback(callback: CancellationCallback | None):
    """Install a long-lived callback, primarily for a process-pool worker."""
    return _callback.set(callback)


def current_cancellation_callback() -> CancellationCallback | None:
    """Return the callback visible in the current execution context."""
    return _callback.get()


def raise_if_cancelled(callback: CancellationCallback | None = None) -> None:
    """Raise immediately when the supplied or context callback requests cancel."""
    requested = _callback.get() if callback is None else callback
    if requested is not None and requested():
        raise SimulationCancelled("simulation cancelled")


def cancellation_checkpoint() -> None:
    """Poll the context callback at an amortized rate in hot loops."""
    callback = _callback.get()
    if callback is None:
        return
    state = _poll_state.get()
    if state is None:
        state = _PollState()
        _poll_state.set(state)
    if state.remaining:
        state.remaining -= 1
        return
    state.remaining = _POLL_INTERVAL
    if callback():
        raise SimulationCancelled("simulation cancelled")


__all__ = [
    "CancellationCallback",
    "SimulationCancelled",
    "cancellation_checkpoint",
    "cancellation_scope",
    "current_cancellation_callback",
    "install_cancellation_callback",
    "raise_if_cancelled",
]
