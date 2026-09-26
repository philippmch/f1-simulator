"""Cached transition stop rows retain the planner's exact eligibility rule."""

import pytest

from f1sim.models import TireCompound, Weather
from f1sim.simulation.rain_strategy import _transition_stop_eligibility_row


def original_may_stop(surface, critical, compound, left, dry, damp):
    """The pre-cache per-offset predicate, kept as an independent test oracle."""
    if critical:
        return True
    limit = (dry if surface.track_wetness < .08 and surface.rain_intensity < .15
             else damp)
    return left > 0 and (
        compound in (TireCompound.INTERMEDIATE, TireCompound.WET)
        or surface.track_wetness > .3
        or limit is None
        or limit > 0
    )


@pytest.mark.parametrize("compound", list(TireCompound))
@pytest.mark.parametrize("left,dry,damp", [
    (0, None, None),
    (1, 0, 0),
    (2, None, 0),
    (1, 0, None),
])
def test_transition_eligibility_row_matches_original_predicate(compound, left, dry, damp):
    surfaces = [
        Weather(track_wetness=wetness, rain_intensity=rain)
        for wetness, rain in [
            (0.0, 0.0),
            (.079999, .149999),
            (.079999, .15),
            (.08, .149999),
            (.08, .15),
            (.3, .149999),
            (.300001, .149999),
            (.3, .15),
            (.300001, .15),
        ]
    ]
    critical = (False, True, False, False, True, False, False, True, False)

    actual = _transition_stop_eligibility_row(
        surfaces, critical, compound, left, dry, damp,
    )

    expected = tuple(
        original_may_stop(surface, critical[offset], compound, left, dry, damp)
        for offset, surface in enumerate(surfaces)
    )
    assert actual == expected
