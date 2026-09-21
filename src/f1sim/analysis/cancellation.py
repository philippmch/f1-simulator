"""Cancellation signals shared by analysis entry points."""


class SimulationCancelled(RuntimeError):
    """Raised when a Monte Carlo run is cancelled before completion."""
