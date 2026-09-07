"""The dashboard must retain classification independently of retirement status."""

import pytest

from f1sim.simulation.race import DriverStatus, RaceResult

pytest.importorskip("fastapi")

from f1sim.web.server import _serialize_race_result  # noqa: E402


@pytest.mark.parametrize("status,laps,classified,eligible", [
    (DriverStatus.FINISHED, 60, True, True),
    (DriverStatus.DNF, 54, True, True),
    (DriverStatus.DNF, 53, False, False),
    (DriverStatus.DNF, None, None, False),
    (DriverStatus.FINISHED, None, None, True),
])
def test_race_payload_preserves_distance_and_classification(status, laps, classified, eligible):
    result = RaceResult("A", "Driver A", "Team", 3, 5000, 0, 1, 90, status,
                        dnf_reason="Engine" if status == DriverStatus.DNF else None,
                        laps_completed=laps, classified=classified)
    payload = _serialize_race_result(result)
    assert payload["status"] == status.value
    assert payload["laps_completed"] == laps
    assert payload["classified"] is eligible
    assert payload["dnf_reason"] == result.dnf_reason
