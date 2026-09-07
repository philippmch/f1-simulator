"""Exercise bounded feed reads through the real HTTP response parser."""

import io
from http.client import HTTPResponse

import pytest

import f1sim.data.current as current_module
from f1sim.data.current import CurrentSeasonDataError, CurrentSeasonDataLoader

URL = "https://example.invalid/live"
LIMIT = 128


class ResponseSocket:
    def __init__(self, wire: bytes) -> None:
        self.stream = io.BytesIO(wire)

    def makefile(self, *_args):
        return self.stream


class TrackedResponse(HTTPResponse):
    def __init__(self, wire: bytes) -> None:
        self.socket = ResponseSocket(wire)
        super().__init__(self.socket)
        self.read_sizes: list[int] = []
        self.bytes_read = 0
        self.begin()

    def read1(self, n: int = -1) -> bytes:
        self.read_sizes.append(n)
        data = super().read1(n)
        self.bytes_read += len(data)
        return data


def install_response(monkeypatch, body: bytes, headers: str = "") -> TrackedResponse:
    response = TrackedResponse(b"HTTP/1.1 200 OK\r\n" + headers.encode() + b"\r\n" + body)
    monkeypatch.setattr(current_module, "MAX_HTTP_RESPONSE_BYTES", LIMIT)
    monkeypatch.setattr(current_module, "HTTP_READ_CHUNK_BYTES", 32)
    monkeypatch.setattr(current_module, "urlopen", lambda *_args, **_kwargs: response)
    return response


def test_declared_oversize_is_rejected_without_reading_body(monkeypatch) -> None:
    response = install_response(monkeypatch, b"", f"Content-Length: {LIMIT + 1}\r\n")
    with pytest.raises(CurrentSeasonDataError, match="exceeds"):
        CurrentSeasonDataLoader._default_http_get(URL, headers={}, timeout=1)
    assert response.read_sizes == []
    assert response.closed
    assert response.socket.stream.closed


@pytest.mark.parametrize("header", ["", "Content-Length: nonsense\r\n"])
def test_oversize_without_valid_length_stops_at_limit_plus_one(monkeypatch, header) -> None:
    response = install_response(monkeypatch, b"x" * (LIMIT * 10), header)
    with pytest.raises(CurrentSeasonDataError, match="exceeds"):
        CurrentSeasonDataLoader._default_http_get(URL, headers={}, timeout=1)
    assert response.bytes_read == LIMIT + 1
    assert all(0 < size <= 32 for size in response.read_sizes)
    assert response.closed


def test_chunked_body_cannot_bypass_limit_with_false_content_length(monkeypatch) -> None:
    body = b"x" * (LIMIT * 10)
    wire_body = f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n"
    response = install_response(
        monkeypatch, wire_body, "Transfer-Encoding: chunked\r\nContent-Length: 1\r\n"
    )
    with pytest.raises(CurrentSeasonDataError, match="exceeds"):
        CurrentSeasonDataLoader._default_http_get(URL, headers={}, timeout=1)
    assert response.bytes_read == LIMIT + 1
    assert response.closed


@pytest.mark.parametrize("declared", [False, True])
def test_exact_boundary_body_is_parsed_and_response_closed(monkeypatch, declared) -> None:
    body = b'{"ok":true}' + b" " * (LIMIT - len(b'{"ok":true}'))
    headers = f"Content-Length: {LIMIT}\r\n" if declared else ""
    response = install_response(monkeypatch, body, headers)
    loader = CurrentSeasonDataLoader()
    assert loader._fetch_json(URL) == {"ok": True}
    assert response.bytes_read == LIMIT
    assert response.closed
    assert loader.provenance["fresh_fetch"] is True


def test_truncated_declared_body_still_fails_even_if_json_prefix_is_valid(monkeypatch) -> None:
    response = install_response(monkeypatch, b"{}", "Content-Length: 100\r\n")
    loader = CurrentSeasonDataLoader()
    with pytest.raises(CurrentSeasonDataError, match="Incomplete"):
        loader._fetch_json(URL)
    assert response.closed
    assert loader.provenance["fresh_fetch"] is False
    assert URL in loader._failed_urls


def test_oversize_marks_failed_provenance_after_earlier_success(monkeypatch) -> None:
    install_response(monkeypatch, b"{}")
    loader = CurrentSeasonDataLoader()
    loader._min_request_interval = 0
    assert loader._fetch_json(URL) == {}
    assert loader.provenance["fresh_fetch"] is True
    install_response(monkeypatch, b"x" * (LIMIT + 1))
    failed_url = URL + "/oversize"
    with pytest.raises(CurrentSeasonDataError, match="exceeds"):
        loader._fetch_json(failed_url)
    assert loader.provenance["fresh_fetch"] is False
    assert loader.provenance["urls"] == [URL]
    assert failed_url in loader._failed_urls


def test_deadline_checked_between_slow_body_reads_and_closes_response(monkeypatch) -> None:
    response = install_response(monkeypatch, b"x" * 20)
    now = [100.0]
    monkeypatch.setattr(current_module.time, "monotonic", lambda: now[0])
    original_read1 = response.read1

    def slow_read1(_size):
        now[0] += 0.25
        return original_read1(1)

    monkeypatch.setattr(response, "read1", slow_read1)
    with pytest.raises(CurrentSeasonDataError, match="read budget exhausted"):
        CurrentSeasonDataLoader._default_http_get(URL, headers={}, timeout=1)
    assert response.bytes_read == 4
    assert response.closed


def test_socket_read_failure_closes_response_and_marks_failed(monkeypatch) -> None:
    response = install_response(monkeypatch, b"{}")

    def fail_read1(_size):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(response, "read1", fail_read1)
    loader = CurrentSeasonDataLoader()
    with pytest.raises(CurrentSeasonDataError, match="request failed"):
        loader._fetch_json(URL)
    assert response.closed
    assert URL in loader._failed_urls
