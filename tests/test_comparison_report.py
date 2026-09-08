"""Comparison reports keep sampling denominators and untrusted labels honest."""

from html.parser import HTMLParser

from f1sim.analysis.montecarlo import DriverStatistics, MonteCarloRunner, SimulationResults
from f1sim.analysis.scenarios import scenario_weather_from_label
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output.comparison import render_comparison_report
from f1sim.simulation.race import DriverStatus, RaceResult


class Document(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.tags = []
        self.text = []
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


def results(stats=None):
    return SimulationResults(1000, "Track", stats or {}, [], [], seed=41)


def test_real_results_report_observed_outcomes_and_paid_stop_denominator():
    result = MonteCarloRunner(
        [Driver(id="A", name="Driver A", team_id="T")],
        {"T": Car(team_id="T", team_name="Team")},
        Track(id="track", name="Track", country="Test", total_laps=3, base_lap_time=90),
        Weather(change_probability=0), seed=41, starting_tires={"A": "hard"},
    ).run(2, parallel=False)
    document = Document(render_comparison_report({"hard": result}, focus_driver="A"))
    text = " ".join(document.text)
    assert "A=hard" in text
    assert "dry; rain 0%; surface wetness 0%; weather change 0%/lap" in text
    assert "(2 recorded races)" in text
    assert "[34.2–100.0%]" in text
    assert "[0.0–65.8%]" in text
    assert any(tag == "details" and "open" in attrs for tag, attrs in document.tags)
    assert not any(tag in {"script", "link", "img", "iframe"} for tag, _ in document.tags)


def test_unequal_observations_use_driver_counts_and_missing_is_not_zero():
    short = results({"A": DriverStatistics(
        "A", "Driver A", "T", wins=1, podiums=2, dnfs=1,
        total_points=30, positions=[1, 3, 20, 9],
    )})
    long = results({"A": DriverStatistics(
        "A", "Driver A", "T", wins=5, podiums=5, dnfs=0,
        total_points=100, positions=[1] * 5 + [10] * 5,
    ), "B": DriverStatistics("B", "Driver B", "T")})
    content = render_comparison_report({"short": short, "long": long})
    assert "25.0%" in content and "[4.6–69.9%]" in content
    assert "50.0%" in content and "[23.7–76.3%]" in content
    assert "<td>7.50</td>" in content and "<td>10.00</td>" in content
    assert content.count("Not recorded (no observed trials)") == 2
    assert content.count("<td>Not recorded</td>") >= 2  # No paid-stop records.
    assert "real-world accuracy or intervals of differences" in content


def test_all_labels_are_literal_text_and_focus_preserves_supplied_order():
    hostile = '<script src="https://bad.test">& hi</script>'
    first = results({"B": DriverStatistics("B", hostile, hostile, positions=[5]),
                     "A": DriverStatistics("A", "A", "T", positions=[5])})
    first.track_name = hostile
    first.race_engine = hostile
    first.input_snapshot = {"starting_tires": {hostile: hostile}}
    second = results({hostile: DriverStatistics(hostile, hostile, hostile, positions=[5])})
    content = render_comparison_report({"z last alphabetically": first, hostile: second},
                                       focus_driver="A")
    document = Document(content)
    assert not any(tag == "script" for tag, _ in document.tags)
    assert hostile in "".join(document.text)
    details = [attrs for tag, attrs in document.tags if tag == "details"]
    assert ["open" in attrs for attrs in details] == [False, True, False]
    assert content.index("z last alphabetically") < content.index("&lt;script")
    assert all("scope" in attrs for tag, attrs in document.tags if tag == "th")


def test_empty_comparison_and_unknown_focus_are_explicit():
    content = render_comparison_report({}, focus_driver="missing")
    assert "No scenarios recorded" in content
    assert "No driver outcomes recorded" in content
    assert "<details" not in content


def test_context_distinguishes_conditions_with_identical_rain_and_surface():
    base = Weather(change_probability=.2)
    scenarios = {}
    for label in ("dry", "cloudy"):
        result = results()
        result.input_snapshot = {
            "weather": scenario_weather_from_label(base, label).weather.model_dump(),
        }
        scenarios[label] = result
    report = render_comparison_report(scenarios)
    assert "dry; rain 0%; surface wetness 0%; weather change 20%/lap" in report
    assert "cloudy; rain 0%; surface wetness 0%; weather change 20%/lap" in report


def test_legacy_starting_context_and_unknown_paid_stops_remain_unknown():
    result = results({"A": DriverStatistics("A", "A", "T", positions=[4])})
    content = render_comparison_report({"legacy": result})
    assert "<td>Not recorded</td>" in content
    assert "<td>0</td><td>Not recorded</td>" in content  # No race distances.
    result.input_snapshot = {"schema_version": 1}
    assert "Automatic" in render_comparison_report({"legacy": result})


def test_distance_comparison_uses_actual_winner_and_separate_denominators():
    def car(position, laps, *, retired=False, limited=False):
        return RaceResult(
            str(position), str(position), "Team", position, 100.0, 0.0, 0, 90.0,
            DriverStatus.DNF if retired else DriverStatus.FINISHED,
            laps_completed=laps, race_time_limited=limited,
        )

    result = results()
    result.race_results = [
        [car(1, 30, limited=True), car(2, 40, retired=True), car(3, 29), car(4, None)],
        [car(1, 50), car(2, 50)],
        [car(1, 10, retired=True)],
    ]
    content = render_comparison_report({"mixed": result})
    assert '40.00 laps <span class="interval">(2 known winners)</span>' in content
    assert content.count("(1 / 3 recorded races)") == 2  # Timed and without a winner.
    assert "(1 / 4 comparable finishers)" in content
    assert "25.0%" in content
    assert "33.3%" in content
    # Requested 1000 simulations must not enter any outcome denominator.
    assert "/ 1000" not in content


def test_legacy_winner_without_distance_does_not_invent_lapping():
    result = results()
    result.race_results = [[RaceResult(
        "A", "A", "Team", 1, 100.0, 0.0, 0, 90.0, DriverStatus.FINISHED,
    )]]
    content = render_comparison_report({"legacy": result})
    assert content.count("(0 / 1 recorded races)") == 2
    assert "comparable finishers)" not in content
    assert "known winners)" not in content


def test_sequences_show_observed_shares_free_fittings_retirements_and_missing():
    result = results({"A": DriverStatistics("A", "A", "Team", positions=[1] * 4)})
    hostile = '<script>bad()</script>'
    for sequence, status in (
        (["soft", "hard", "hard"], DriverStatus.FINISHED),
        (["soft", "hard", "hard"], DriverStatus.DNF),
        ([hostile], DriverStatus.FINISHED),
        ([], DriverStatus.FINISHED),
    ):
        result.race_results.append([RaceResult(
            "A", "A", "Team", 1, 100.0, 0.0, 0, 90.0, status, strategy=sequence,
        )])
    content = render_comparison_report({"mixed": result})
    assert 'soft → hard → hard</td><td>2 / 3 (66.7%)</td><td>1</td><td>1</td>' in content
    assert "1 / 3 (33.3%)" in content
    assert "Not recorded (missing sequence in 1 race)" in content
    assert hostile in "".join(Document(content).text)
    assert not any(tag == "script" for tag, _ in Document(content).tags)
    assert "<td>0.00 " in content  # Repeated fittings did not become paid stops.
