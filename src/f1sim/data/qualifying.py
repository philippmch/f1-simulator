"""Explicit qualifying classification and timing validity without changing legacy rows.

Provider-only ranks are not a status check: some disqualified entrants retain
ordinary (occasionally duplicate) positions. Official recovery is deliberately
limited to an ambiguous classification; a unique provider permutation is not
claimed to have been checked for disqualifications.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from urllib.parse import urlparse

QUALIFYING_CLASSIFICATION = "qualifying_classification"
STAGES = ("Q1", "Q2", "Q3")
_STATUSES = {"classified", "disqualified", "not_classified", "did_not_start",
             "did_not_qualify", "untimed"}
_MARKERS = {"DQ": "disqualified", "DSQ": "disqualified", "NC": "not_classified",
            "DNS": "did_not_start", "DNP": "did_not_start", "DNQ": "did_not_qualify",
            "DNF": "not_classified"}


def _constructor_key(label):
    # Qualifying tables append engine suppliers. Recognize the chassis before
    # Ferrari/Mercedes/Renault, which also appear as independent constructors.
    text = re.sub(r"[\W_]+", " ", str(label).lower()).strip()
    aliases = (("red bull", "red_bull"), ("mclaren", "mclaren"), ("haas", "haas"),
               ("sauber", "sauber"), ("alfa romeo", "alfa"), ("alfa", "alfa"),
               ("racing bulls", "rb"), ("rb", "rb"),
               ("alphatauri", "alphatauri"), ("alpha tauri", "alphatauri"),
               ("toro rosso", "toro_rosso"), ("aston martin", "aston_martin"),
               ("alpine", "alpine"), ("williams", "williams"),
               ("racing point", "racing_point"), ("force india", "force_india"),
               ("lotus", "lotus"), ("caterham", "caterham"), ("marussia", "marussia"),
               ("manor", "manor"), ("audi", "audi"), ("cadillac", "cadillac"),
               ("ferrari", "ferrari"), ("mercedes", "mercedes"), ("renault", "renault"))
    return next((team for alias, team in aliases
                 if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text)), text)


def _identity(row):
    driver = row.get("Driver", row.get("driver"))
    constructor = row.get("Constructor", row.get("team"))
    return {"driver": driver.get("driverId") if isinstance(driver, Mapping) else driver,
            "code": driver.get("code") if isinstance(driver, Mapping) else row.get("code"),
            "team": (constructor.get("constructorId") or constructor.get("name"))
            if isinstance(constructor, Mapping) else constructor}


def _note_text(value):
    text = "".join(c for c in unicodedata.normalize("NFKD", str(value))
                   if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _note_labels(identity):
    name = _note_text(identity["driver"])
    labels = {name, _note_text(identity["code"])}
    if name and len(name.split()[-1]) >= 4:
        labels.add(name.split()[-1])
    return labels - {""}


def _seconds(value):
    """Canonical comparison of retained strings and normalized numeric seconds."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("time") or value.get("Time")
    try:
        parts = str(value).strip().split(":")
        if not 1 <= len(parts) <= 3 or any(not re.fullmatch(r"\d+(?:\.\d+)?", p)
                                          for p in parts):
            return None
        if len(parts) > 1 and float(parts[-1]) >= 60:
            return None
        number = sum(float(part) * 60 ** i for i, part in enumerate(reversed(parts)))
        return number if math.isfinite(number) and number > 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def _rank(value):
    return int(value) if not isinstance(value, bool) and str(value).isdecimal() else None


def classification(row: Mapping) -> dict | None:
    """Validate optional metadata; absence retains the original source semantics."""
    if QUALIFYING_CLASSIFICATION not in row:
        return None
    value = row[QUALIFYING_CLASSIFICATION]
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Unsupported qualifying classification metadata")
    status, rank = value.get("status"), value.get("position")
    if status not in _STATUSES:
        raise ValueError("Unknown qualifying classification status")
    if ((status == "classified" and (type(rank) is not int or rank < 1))
            or (status != "classified" and rank is not None)):
        raise ValueError("Qualifying status and classification rank disagree")
    valid, times = value.get("stage_validity"), value.get("stage_times")
    if (not isinstance(valid, dict) or set(valid) != set(STAGES)
            or not isinstance(times, dict) or set(times) != set(STAGES)):
        raise ValueError("Qualifying metadata requires independent stage validity")
    for stage in STAGES:
        time = times[stage]
        if type(valid[stage]) is not bool or (valid[stage] != (time is not None)):
            raise ValueError("Qualifying stage time and validity disagree")
        if time is not None and (isinstance(time, bool) or not isinstance(time, (int, float))
                                 or not math.isfinite(time) or not 30 <= time <= 240):
            raise ValueError("Qualifying stage times must be finite seconds in the source range")
        if status in {"disqualified", "did_not_start", "untimed"} and valid[stage]:
            raise ValueError("Invalid qualifying laps cannot supply pace")
    source = value.get("source")
    if (not isinstance(source, dict) or source.get("session") != "Qualifying"
            or type(source.get("year")) is not int or type(source.get("round")) is not int
            or source["year"] < 1950 or source["round"] < 1
            or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("content_sha256", "")))):
        raise ValueError("Qualifying metadata lacks primary event/session evidence")
    url = urlparse(str(source.get("source_url", "")))
    if (url.scheme != "https" or url.netloc != "www.formula1.com"
            or not re.fullmatch(rf"/en/results/{source['year']}/races/\d+/[a-z0-9-]+/qualifying",
                                url.path) or url.query or url.fragment):
        raise ValueError("Qualifying metadata requires an official GP qualifying source")
    if not isinstance(value.get("reported"), dict) or not isinstance(value.get("official"), dict):
        raise ValueError("Qualifying metadata must retain reported and official fields")
    identity = value.get("identity")
    if (not isinstance(identity, dict) or set(identity) != {"driver", "code", "team"}
            or any(not isinstance(item, str) or not item.strip() for item in identity.values())
            or identity != _identity(row)):
        raise ValueError("Qualifying metadata driver/constructor identity differs")
    for key in ("year", "round"):
        if key in row and str(row[key]) != str(source[key]):
            raise ValueError("Qualifying metadata event identity differs")
    reported, official = value["reported"], value["official"]
    raw_rank = row.get("qualifying_position", row.get("position"))
    if _rank(reported.get("position")) != _rank(raw_rank):
        raise ValueError("Retained reported qualifying rank differs from source row")
    for stage in STAGES:
        if _seconds(reported.get(stage)) != _seconds(row.get(stage, row.get(stage.lower()))):
            raise ValueError("Retained reported qualifying time differs from source row")
    if "note" in official:
        note = str(official["note"])
        matched = official.get("matched_label")
        if (status != "did_not_start" or rank is not None or any(times.values())
                or not re.search(r"\bdid not (?:take part|participate) in qualifying\b", note, re.I)
                or matched not in _note_labels(identity)
                or not re.search(rf"(?<!\w){re.escape(matched)}(?!\w)", _note_text(note))):
            raise ValueError("Qualifying absence lacks matching explicit official note")
    else:
        display = str(official.get("position", "")).strip().upper()
        derived_status = "classified" if display.isdecimal() else _MARKERS.get(display)
        derived_rank = int(display) if display.isdecimal() else None
        if status != derived_status or rank != derived_rank:
            raise ValueError("Effective qualifying classification differs from official display")
        if (str(official.get("driver", "")).split()[-1:] != [identity["code"]]
                or (_rank(official.get("number")) or 0) < 1):
            raise ValueError("Official qualifying row identity is incomplete")
        if _constructor_key(official.get("team", "")) != _constructor_key(identity["team"]):
            raise ValueError("Official qualifying constructor differs from source row")
        if "number" in row and _rank(row["number"]) != _rank(official["number"]):
            raise ValueError("Retained official qualifying car number differs")
        for stage in STAGES:
            expected = (_seconds(official.get(stage)) if status not in
                        {"disqualified", "did_not_start"} else None)
            if times[stage] != expected:
                raise ValueError("Effective qualifying time differs from official stage evidence")
    return value


def effective_qualifying(row: Mapping, *, year: int, round_number: int) -> dict | None:
    """Return normalized usable values, or None for an untouched legacy row."""
    value = classification(row)
    if value is None:
        return None
    if (type(year) is not int or type(round_number) is not int
            or value["source"]["year"] != year or value["source"]["round"] != round_number):
        raise ValueError("Qualifying metadata event identity differs")
    return {"qualifying_position": value["position"],
            **{stage.lower(): value["stage_times"][stage] for stage in STAGES}}


def qualifying_rank(row: Mapping, legacy):
    value = classification(row)
    return value["position"] if value is not None else legacy


def qualifying_stage(row: Mapping, stage: str, legacy):
    value = classification(row)
    return value["stage_times"][stage.upper()] if value is not None else legacy


def complete_classification(rows: Sequence[Mapping], *, field_size: int | None = None) -> bool:
    """A whole field may contain explicitly unclassified entrants, never guessed ones."""
    n = len(rows) if field_size is None else field_size
    if n < 2 or len(rows) != n:
        return False
    ranks, explicit, sources = [], False, set()
    for row in rows:
        meta = classification(row)
        explicit |= meta is not None
        if meta is not None:
            source = meta["source"]
            sources.add((source["year"], source["round"], source["source_url"],
                         source["content_sha256"]))
            rank = meta["position"]
            if rank is None:
                continue  # A validated, explicit non-classified status.
        else:
            raw = row.get("qualifying_position", row.get("position"))
            if isinstance(raw, bool) or not str(raw).isdecimal():
                return False
            rank = int(raw)
        if not 1 <= rank <= n or rank in ranks:
            return False
        ranks.append(rank)
    if explicit:
        # Partial primary classification is one complete joined source, not an
        # explicit exception grafted onto unverified provider-only ranks.
        return len(sources) == 1 and all(QUALIFYING_CLASSIFICATION in row for row in rows)
    return sorted(ranks) == list(range(1, n + 1))


def parse_official_classification(text, *, loader, year, event, rows, source_url):
    """Join primary numeric/DQ/NC rows, or an explicit no-participation note.

    The supplied rows define the entrant identities and constructors. Missing
    qualifying evidence never creates an entrant or invents a disqualification.
    Original provider fields remain intact and are also retained in metadata.
    """
    from f1sim.data.current import _as_int, _first, _normalise_text, _parse_time_seconds
    from f1sim.data.practice import _OFFICIAL_SLUGS, _ResultsTable

    class Table(_ResultsTable):
        def __init__(self):
            super().__init__()
            self.paragraph, self.notes = None, []

        def handle_starttag(self, tag, attrs):
            super().handle_starttag(tag, attrs)
            if tag == "p":
                self.paragraph = []

        def handle_data(self, data):
            super().handle_data(data)
            if self.paragraph is not None:
                self.paragraph.append(data)

        def handle_endtag(self, tag):
            super().handle_endtag(tag)
            if tag == "p" and self.paragraph is not None:
                self.notes.append(" ".join(" ".join(self.paragraph).split()))
                self.paragraph = None

    table = Table()
    table.feed(text)
    event_name = re.split(r"\s+grand\s+prix\b", str(event.get("race", "")), flags=re.I)[0]
    names = {_normalise_text(event_name)} if event_name else set()
    expected_slug = _OFFICIAL_SLUGS.get(event.get("circuit_id"))
    if expected_slug is not None:
        if urlparse(source_url).path.split("/")[-2:-1] != [expected_slug]:
            raise ValueError("Official qualifying URL differs from the calendar circuit")
        names.add(_normalise_text(expected_slug.replace("-", " ")))
    if not event_name or not any(
        str(year) in heading and "QUALIFYING" in heading.upper()
        and "SPRINT" not in heading.upper()
        and any(name in _normalise_text(heading) for name in names)
        for heading in table.headings
    ):
        raise ValueError("Official qualifying event or session differs")
    header = ["pos", "no", "driver", "team", "q1", "q2", "q3", "laps"]
    if not any([c.lower().rstrip(".") for c in cells] == header for cells, _ in table.rows):
        raise ValueError("Official qualifying page lacks explicit Q1/Q2/Q3 columns")
    expected = {}
    for index, row in enumerate(rows):
        code = str(_first(loader._extract_driver(row), "code", "Code", "abbreviation",
                          default="")).strip().upper()
        if not code or code in expected:
            raise ValueError("Qualifying entrants have ambiguous driver codes")
        expected[code] = index
    if not 2 <= len(expected) <= 30:
        raise ValueError("Official qualifying requires the complete known entrant field")
    source = {"source_url": source_url, "year": year, "round": int(event["round"]),
              "session": "Qualifying", "content_sha256": hashlib.sha256(text.encode()).hexdigest()}
    output, seen, numbers = copy.deepcopy(rows), set(), set()

    def save(index, status, rank, times, official):
        row = rows[index]
        output[index][QUALIFYING_CLASSIFICATION] = {
            "schema_version": 1, "status": status, "position": rank,
            "identity": _identity(row),
            "stage_times": times, "stage_validity": {stage: times[stage] is not None
                                                      for stage in STAGES},
            "reported": {"position": row.get("position"),
                         **{stage: row.get(stage, row.get(stage.lower())) for stage in STAGES}},
            "official": official, "source": copy.deepcopy(source),
        }
        classification(output[index])

    for cells, _ in table.rows:
        if len(cells) != 8 or cells[0].lower().rstrip(".") == "pos":
            continue
        display = cells[0].strip().upper()
        if not display.isdecimal() and display not in _MARKERS:
            continue
        code = cells[2].split()[-1].upper() if cells[2].split() else ""
        if code not in expected or code in seen:
            raise ValueError("Official qualifying driver identity differs")
        index, row = expected[code], rows[expected[code]]
        number, reported_number = _as_int(cells[1]), _as_int(row.get("number"))
        if (number is None or number < 1 or number in numbers
                or (reported_number is not None and number != reported_number)):
            raise ValueError("Official qualifying car number conflicts")
        numbers.add(number)
        if _constructor_key(_identity(row)["team"]) != _constructor_key(cells[3]):
            raise ValueError("Official qualifying constructor conflicts")
        status = "classified" if display.isdecimal() else _MARKERS[display]
        times = {}
        for stage, raw in zip(STAGES, cells[4:7], strict=True):
            value = None
            if raw and raw.strip().upper() not in {"-", "—", "DNF", "DNS", "DQ", "DNQ"}:
                value = _parse_time_seconds(raw)
                if value is None or not math.isfinite(value) or not 30 <= value <= 240:
                    raise ValueError("Invalid official qualifying stage time")
            times[stage] = None if status in {"disqualified", "did_not_start"} else value
        save(index, status, int(display) if display.isdecimal() else None, times,
             {"position": cells[0], "number": cells[1], "driver": cells[2], "team": cells[3],
              **dict(zip(STAGES, cells[4:7], strict=True))})
        seen.add(code)
    for code in set(expected) - seen:
        row = rows[expected[code]]
        labels = {label for label in _note_labels(_identity(row)) if sum(
            label in _note_labels(_identity(other)) for other in rows) == 1}
        matches = [(note, label) for paragraph in table.notes
                   for note in re.split(r"(?<=[.!?;])\s+", paragraph)
                   for label in sorted(labels)
                   if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", _note_text(note))
                   and re.search(r"\bdid not (?:take part|participate) in qualifying\b",
                                 note, re.I)]
        if not matches:
            raise ValueError(
                "Official qualifying omits an entrant without explicit status evidence",
            )
        note, label = matches[0]
        save(expected[code], "did_not_start", None, dict.fromkeys(STAGES),
             {"note": note, "matched_label": label})
        seen.add(code)
    if seen != set(expected) or not complete_classification(output):
        raise ValueError("Official qualifying classification is incomplete or ambiguous")
    return output
