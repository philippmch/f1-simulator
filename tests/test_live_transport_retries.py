"""Fresh transport retries honor server delays, deadlines, and feed provenance."""

import io
import json
import ssl
import threading
from datetime import datetime, timedelta, timezone
from email.message import Message
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError

import pytest
from test_live_transport import TrackedResponse

from f1sim.data import CurrentSeasonDataError, CurrentSeasonDataLoader
from f1sim.data import current as data

URL = "https://example.invalid/current"


class Clock:
    def __init__(self, monkeypatch):
        self.now = 100.0
        self.sleeps = []
        self.utc = datetime(datetime.now(timezone.utc).year, 10, 4, 12, tzinfo=timezone.utc)
        monkeypatch.setattr(data.time, "monotonic", lambda: self.now)
        monkeypatch.setattr(data.time, "sleep", self.sleep)
        monkeypatch.setattr(data, "_utc_now", lambda: self.utc + timedelta(seconds=self.now - 100))

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _error(status, after=None):
    headers = Message()
    if after is not None:
        headers["Retry-After"] = after
    return HTTPError(URL, status, "provider response", headers, io.BytesIO(b"busy"))


def _response(body=b'{"ok":true}'):
    return TrackedResponse(b"HTTP/1.1 200 OK\r\nContent-Length: "
                           + str(len(body)).encode() + b"\r\n\r\n" + body)


def _install(monkeypatch, outcomes):
    calls = []
    pending = iter(outcomes)

    def open_request(request, *, timeout):
        calls.append((request, timeout))
        outcome = next(pending)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(data, "urlopen", open_request)
    return calls


@pytest.mark.parametrize("date_header", [False, True])
def test_rate_limited_get_waits_and_returns_only_fresh_current_data(monkeypatch, date_header):
    clock = Clock(monkeypatch)
    after = format_datetime(clock.utc + timedelta(seconds=10), usegmt=True) if date_header else "10"
    error = _error(429, after)
    response = _response()
    calls = _install(monkeypatch, [error, response])
    loader = CurrentSeasonDataLoader(timeout=20, fetch_budget=20)
    assert loader._fetch_json(URL) == {"ok": True}
    assert clock.sleeps == [10]
    assert [timeout for _, timeout in calls] == [20, 10]
    assert all(request.full_url == URL for request, _ in calls)
    assert all("no-cache" in request.get_header("Cache-control") for request, _ in calls)
    assert error.fp.closed and response.closed
    provenance = loader.get_provenance()
    assert provenance["fresh_fetch"] and provenance["urls"] == [URL]
    assert provenance["http_retries"][0]["http_status"] == 429
    assert provenance["http_retries"][0]["delay_seconds"] == 10
    provenance["http_retries"][0]["url"] = "changed"
    assert loader.get_provenance()["http_retries"][0]["url"] == URL
    loader.refresh()
    assert "http_retries" not in loader.provenance and loader.provenance["urls"] == []


@pytest.mark.parametrize("after", ["20", "1000000000000000000000000000000"])
def test_server_wait_that_exceeds_request_budget_never_retries_or_sleeps(monkeypatch, after):
    clock = Clock(monkeypatch)
    error = _error(429, after)
    calls = _install(monkeypatch, [error])
    loader = CurrentSeasonDataLoader(timeout=20)
    with pytest.raises(CurrentSeasonDataError) as failed:
        loader._fetch_json(URL)
    assert failed.value.__cause__ is error and error.fp.closed
    assert len(calls) == 1 and clock.sleeps == []
    assert not loader.provenance["fresh_fetch"]
    assert "http_retries" not in loader.provenance


def test_retry_waits_consume_shared_fetch_budget_before_another_page(monkeypatch):
    clock = Clock(monkeypatch)
    calls = _install(monkeypatch, [_error(429, "1"), _response()])
    loader = CurrentSeasonDataLoader(timeout=20, fetch_budget=2)
    assert loader._fetch_json(URL) == {"ok": True}
    assert [timeout for _, timeout in calls] == [2, 1]
    clock.now = loader._fetch_budget_deadline
    with pytest.raises(CurrentSeasonDataError, match="fetch budget exhausted"):
        loader._fetch_json(URL + "/next")
    assert len(calls) == 2 and not loader.provenance["fresh_fetch"]


def test_next_page_is_paced_from_latest_retry_attempt(monkeypatch):
    clock = Clock(monkeypatch)
    _install(monkeypatch, [_error(429, "1"), _response(), _response()])
    loader = CurrentSeasonDataLoader()
    assert loader._fetch_json(URL) == {"ok": True}
    retried_at = clock.now
    assert loader._fetch_json(URL + "/next") == {"ok": True}
    assert clock.sleeps == [1, loader._min_request_interval]
    assert clock.now - retried_at == pytest.approx(loader._min_request_interval)


def test_repeated_throttling_is_bounded_and_closes_every_error_response(monkeypatch):
    clock = Clock(monkeypatch)
    errors = [_error(429, "0") for _ in range(3)]
    calls = _install(monkeypatch, errors)
    loader = CurrentSeasonDataLoader()
    with pytest.raises(CurrentSeasonDataError) as failed:
        loader._fetch_json(URL)
    assert failed.value.__cause__ is errors[-1]
    assert len(calls) == 3 and clock.sleeps == [.5, 1]
    assert all(error.fp.closed for error in errors)
    assert not loader.provenance["fresh_fetch"] and loader.provenance["urls"] == []
    assert len(loader.provenance["http_retries"]) == 2


def test_expiry_during_retry_wait_does_not_record_or_send_another_attempt(monkeypatch):
    clock = Clock(monkeypatch)
    error = _error(429, "1")
    calls = _install(monkeypatch, [error])
    monkeypatch.setattr(data.time, "sleep", lambda _seconds: setattr(clock, "now", 103))
    loader = CurrentSeasonDataLoader(timeout=2)
    with pytest.raises(CurrentSeasonDataError, match="budget exhausted before retry"):
        loader._fetch_json(URL)
    assert len(calls) == 1 and "http_retries" not in loader.provenance


@pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
def test_transient_http_status_retries_without_a_server_delay(monkeypatch, status):
    clock = Clock(monkeypatch)
    error = _error(status)
    error.headers = None
    calls = _install(monkeypatch, [error, _response()])
    assert CurrentSeasonDataLoader()._fetch_json(URL) == {"ok": True}
    assert len(calls) == 2 and clock.sleeps == [.5]


@pytest.mark.parametrize("error", [
    _error(401), _error(403), _error(404), _error(410), _error(501),
    URLError(ssl.SSLCertVerificationError("untrusted certificate")), URLError("bad host"),
])
def test_permanent_http_and_certificate_failures_do_not_retry(monkeypatch, error):
    clock = Clock(monkeypatch)
    calls = _install(monkeypatch, [error])
    loader = CurrentSeasonDataLoader()
    with pytest.raises(CurrentSeasonDataError) as failed:
        loader._fetch_json(URL)
    assert failed.value.__cause__ is error
    assert len(calls) == 1 and clock.sleeps == []


def test_connection_reset_discards_partial_body_before_fresh_response(monkeypatch):
    clock = Clock(monkeypatch)
    partial = _response()
    reads = [0]

    def broken_read(_size):
        reads[0] += 1
        if reads[0] == 1:
            return b'{"partial":true}'
        raise ConnectionResetError("connection reset during body")

    monkeypatch.setattr(partial, "read1", broken_read)
    complete = _response()
    calls = _install(monkeypatch, [partial, complete])
    assert CurrentSeasonDataLoader()._fetch_json(URL) == {"ok": True}
    assert len(calls) == 2 and clock.sleeps == [.5]
    assert partial.closed and complete.closed


@pytest.mark.parametrize("body,message", [
    (b"not JSON", "Invalid JSON"), (b"\xff", "Invalid UTF-8"),
])
def test_invalid_fresh_payload_is_not_hidden_by_another_retry(monkeypatch, body, message):
    clock = Clock(monkeypatch)
    calls = _install(monkeypatch, [_error(503), _response(body)])
    loader = CurrentSeasonDataLoader()
    with pytest.raises(CurrentSeasonDataError, match=message):
        loader._fetch_json(URL)
    assert len(calls) == 2 and clock.sleeps == [.5]
    assert not loader.provenance["fresh_fetch"]


def test_custom_getter_failure_remains_an_explicit_integration_boundary(monkeypatch):
    clock = Clock(monkeypatch)
    calls = []

    def custom(url):
        calls.append(url)
        raise _error(429, "1")

    loader = CurrentSeasonDataLoader(http_getter=custom)
    with pytest.raises(CurrentSeasonDataError):
        loader._fetch_json(URL)
    assert calls == [URL] and clock.sleeps == []


def test_fresh_retry_does_not_accept_another_season(monkeypatch):
    Clock(monkeypatch)
    loader = CurrentSeasonDataLoader()
    body = json.dumps({"MRData": {"RaceTable": {"season": str(loader.current_year - 1),
                                               "Races": []}, "total": "0"}}).encode()
    calls = _install(monkeypatch, [_error(503), _response(body)])
    with pytest.raises(CurrentSeasonDataError, match="expected current UTC season"):
        loader._fetch_paginated(URL, ("Races",), expected_season=loader.current_year)
    assert len(calls) == 2 and not loader.provenance["fresh_fetch"]


def test_real_calendar_fetch_recovers_from_throttle_and_retains_fresh_provenance(monkeypatch):
    test_client = pytest.importorskip("fastapi.testclient")
    year = datetime.now(timezone.utc).year
    payload = json.dumps({"MRData": {"total": "1", "RaceTable": {
        "season": str(year), "Races": [{"season": str(year), "round": "1",
                                        "raceName": "Live transport test", "date": f"{year}-12-01",
                                        "Circuit": {"circuitId": "test", "circuitName": "Test",
                                                    "Location": {"country": "Test"}}}],
    }}}).encode()
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append((self.path, self.headers.get("Cache-Control")))
            if len(received) == 1:
                self.send_response(429)
                self.send_header("Retry-After", "0")
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        loader = CurrentSeasonDataLoader()
        loader.JOLPICA_BASE_URL = f"http://127.0.0.1:{server.server_port}/ergast/f1"
        events = loader.list_available_events(year)
        assert len(events) == 1 and events[0]["race"] == "Live transport test"
        assert loader.provenance["fresh_fetch"] and loader.provenance["cache"] == "disabled"
        assert loader.provenance["http_retries"][0]["http_status"] == 429
        assert len(received) == 2 and received[0] == received[1]
        assert "no-cache" in received[0][1]
        from f1sim.web import server as web

        monkeypatch.setattr(web, "_get_loader", lambda: loader)
        with test_client.TestClient(web.build_fastapi_app()) as client:
            response = client.get(f"/api/calendar?year={year}")
        assert response.status_code == 200
        result = response.json()
        assert result["events"][0]["race"] == "Live transport test"
        assert result["provenance"]["fresh_fetch"]
        assert result["provenance"]["http_retries"][0]["http_status"] == 429
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
