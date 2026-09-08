"""Comparison reports keep sampling denominators and untrusted labels honest."""

from html.parser import HTMLParser

from f1sim.analysis.montecarlo import DriverStatistics, MonteCarloRunner, SimulationResults
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output.comparison import render_comparison_report


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
    assert "Rain 0%; surface wetness 0%" in text
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


def test_legacy_starting_context_and_unknown_paid_stops_remain_unknown():
    result = results({"A": DriverStatistics("A", "A", "T", positions=[4])})
    content = render_comparison_report({"legacy": result})
    assert content.count("Not recorded") == 3  # Starting choice, weather and paid stops.
    result.input_snapshot = {"schema_version": 1}
    assert "Automatic" in render_comparison_report({"legacy": result})
