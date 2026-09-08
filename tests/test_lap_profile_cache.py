"""Cached static pace terms must track edits to mutable scenario models."""

import pytest

from f1sim.models import Car, Track
from f1sim.models.track import Sector
from f1sim.simulation.lap import LapSimulator


def track():
    return Track(id="t", name="Test", country="Test", total_laps=20, base_lap_time=90,
                 sectors=[Sector(number=1, base_time=10, is_high_speed=True,
                                 overtake_opportunity=.8),
                          Sector(number=2, base_time=30, overtake_opportunity=.2)])


def test_sector_profile_follows_nested_mutations_and_replacement():
    venue = track()
    assert LapSimulator._track_profile(venue) == pytest.approx((.25, .35))
    venue.sectors[0].base_time = 30
    assert LapSimulator._track_profile(venue) == pytest.approx((.5, .5))
    venue.sectors[1].is_high_speed = True
    assert LapSimulator._track_profile(venue) == pytest.approx((1, .5))
    venue.sectors[0].overtake_opportunity = .4
    assert LapSimulator._track_profile(venue) == pytest.approx((1, .3))
    venue.sectors = []
    venue.overtake_difficulty = .75
    assert LapSimulator._track_profile(venue) == (0, .25)
    venue.overtake_difficulty = .25
    assert LapSimulator._track_profile(venue) == (0, .75)
    venue.sectors = track().sectors
    assert LapSimulator._track_profile(venue) == pytest.approx((.25, .35))


def test_car_term_follows_ratings_reference_time_and_track_edits():
    venue = track()
    car = Car(team_id="T", team_name="Team", downforce_level=.8, straight_line_speed=.8)
    assert LapSimulator._track_car_delta(car, venue, 100) == 0
    car.downforce_level = 1
    assert LapSimulator._track_car_delta(car, venue, 100) == pytest.approx(-.06)
    car.straight_line_speed = 1
    assert LapSimulator._track_car_delta(car, venue, 100) == pytest.approx(-.1072)
    assert LapSimulator._track_car_delta(car, venue, 200) == pytest.approx(-.2144)
    venue.sectors[1].is_high_speed = True
    assert LapSimulator._track_car_delta(car, venue, 100) == pytest.approx(.0068)
    # A different model with the same ID must not inherit the edited car/track term.
    assert LapSimulator._track_car_delta(
        Car(team_id="T", team_name="Team"), track(), 100,
    ) == 0
