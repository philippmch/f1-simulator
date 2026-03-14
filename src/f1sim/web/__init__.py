"""Web dashboard module."""

from .dashboard import (
    DashboardRunRequest,
    build_dashboard_html,
    build_fastapi_app,
    run_dashboard_simulation,
)

__all__ = [
    "DashboardRunRequest",
    "build_dashboard_html",
    "build_fastapi_app",
    "run_dashboard_simulation",
]
