"""Validate the dry schedule diagnostic's optional fitting-cost interface."""

import runpy
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def diagnostic():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "check_dry_pit_schedules.py")
    )


@pytest.mark.parametrize("spec", [
    "soft=nan",
    "soft=inf",
    "soft=60.01",
    "soft=0.5,soft=1",
    "ultra=1",
    "soft",
])
def test_invalid_warmup_cli_values_stop_before_schedule_enumeration(
    diagnostic, monkeypatch, spec,
):
    monkeypatch.setitem(
        diagnostic["main"].__globals__, "compare_schedules",
        lambda **kwargs: pytest.fail("invalid input reached schedule comparison"),
    )

    with pytest.raises(SystemExit) as error:
        diagnostic["main"](["--tire-warmup", spec])

    assert error.value.code == 2


def test_cli_passes_parsed_profile_to_comparison(diagnostic, monkeypatch, capsys):
    captured = {}

    def compare_schedules(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setitem(
        diagnostic["main"].__globals__, "compare_schedules", compare_schedules,
    )
    diagnostic["main"]([
        "--engine", "standard", "--tire-warmup", "soft=0.5,hard=1",
    ])

    assert captured == {
        "engines": ("standard",),
        "tire_warmup": {"soft": 0.5, "hard": 1.0},
    }
    assert capsys.readouterr().out == "[]\n"


def test_comparison_records_profile_for_adaptive_and_every_fixed_schedule(
    diagnostic, monkeypatch,
):
    case = {
        "name": "small_profile_check",
        "laps": 3,
        "base_lap_time": 90,
        "pit_lane_delta": 22,
        "tire_stress": 0.7,
        "degradation": 1.0,
    }
    profile = {"soft": 0.5, "hard": 1.0}
    observed = []
    run_race = diagnostic["run_race"]

    def recording_run(*args, **kwargs):
        observed.append(kwargs.get("tire_warmup"))
        return run_race(*args, **kwargs)

    monkeypatch.setitem(
        diagnostic["compare_schedules"].__globals__, "run_race", recording_run,
    )
    row = diagnostic["compare_schedules"](
        cases=(case,), engines=("standard",), tire_warmup=profile,
    )[0]

    canonical = {"hard": 1.0, "soft": 0.5}
    assert row["tire_warmup"] == canonical
    assert row["tire_warmup_policy"] == "post_fit_first_lap_v1"
    assert row["schedules_checked"] == len(list(diagnostic["schedules"](case["laps"])))
    assert len(observed) == row["schedules_checked"] + 1
    assert observed == [canonical] * len(observed)


def test_default_and_all_zero_profiles_keep_legacy_comparison_rows(diagnostic):
    case = {
        "name": "small_legacy_check",
        "laps": 3,
        "base_lap_time": 90,
        "pit_lane_delta": 22,
        "tire_stress": 0.7,
        "degradation": 1.0,
    }

    default = diagnostic["compare_schedules"](cases=(case,), engines=("standard",))
    zero_profile = diagnostic["compare_schedules"](
        cases=(case,), engines=("standard",), tire_warmup={"soft": 0.0, "hard": 0},
    )

    assert default == zero_profile
    assert "tire_warmup" not in default[0]
    assert "tire_warmup_policy" not in default[0]
