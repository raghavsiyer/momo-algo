"""Float through Firecrawl (market_data/firecrawl_float.py), against canned answers.

Nothing is sent anywhere: the client gets a fake sender and a temporary cache.
The answers are trimmed from Firecrawl's on 2026-11-16, the same ones
ScalpFeed's harness uses, so both read the cache the same way.
"""

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from market_data import firecrawl_float as ff


def _stats(symbol, flt, shares, volume):
    return json.dumps({"success": True, "data": {"alexandria": [{
        "provider": "stockanalysis-com", "capability": "securities/stock_statistics", "creditsCost": 5,
        "data": {"name": symbol, "symbol": symbol, "sections": [
            {"id": "shares", "items": [
                {"hover": shares, "id": "sharesout", "title": "Shares Outstanding", "value": "x"},
                {"hover": flt, "id": "float", "title": "Float", "value": "x"}]},
            {"id": "stockPrice", "items": [
                {"hover": volume, "id": "averageVolume", "title": "Average Volume (20 Days)", "value": "x"}]},
        ]}}]}})


NOT_FOUND = json.dumps({"success": True, "data": {"alexandria": [{
    "provider": "stockanalysis-com", "capability": "securities/stock_statistics",
    "error": {"code": "not_found", "message": "The workflow could not find that resource.", "status": 404}}]}})


class _Fake:
    def __init__(self):
        self.now = datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc)
        self.sent: list[str] = []
        self.answers: dict[str, tuple[int, str]] = {}

    def send(self, key, body):
        symbol = body["alexandria"]["options"]["symbol"]
        self.sent.append(symbol)
        return self.answers.get(symbol, (200, _stats(symbol, "1,000,000", "2,000,000", "500,000")))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    key_path = tmp_path / "firecrawl.json"
    key_path.write_text(json.dumps({"apiKey": "fc-test"}))
    fake = _Fake()
    client = ff.FirecrawlFloatClient(
        cache_path=str(tmp_path / "float-cache.json"), key_path=str(key_path),
        send=fake.send, utc_now=lambda: fake.now,
    )
    return client, fake, tmp_path


def test_reads_float_shares_and_volume_from_the_exact_values(setup):
    client, fake, _ = setup
    fake.answers["GDHG"] = (200, _stats("GDHG", "29,519,326", "47,127,140", "5,256,126"))
    fd = client.get_float("GDHG")
    assert fd.float_shares == 29_519_326 and fd.outstanding_shares == 47_127_140
    assert fd.float_shares_m == pytest.approx(29.519326) and fd.source == "firecrawl"
    assert client.facts("GDHG")["averageVolume"] == 5_256_126


def test_no_key_sends_nothing(setup):
    client, fake, tmp_path = setup
    (tmp_path / "firecrawl.json").write_text(json.dumps({"apiKey": ""}))
    assert client.get_float("GDHG") is None and fake.sent == []


def test_cache_shared_and_kept_a_week(setup):
    client, fake, tmp_path = setup
    client.get_float("GDHG")
    client.get_float("GDHG")
    assert fake.sent == ["GDHG"]
    # A second client (another process) reads the same file instead of paying again.
    other = ff.FirecrawlFloatClient(cache_path=str(tmp_path / "float-cache.json"),
                                    key_path=str(tmp_path / "firecrawl.json"),
                                    send=fake.send, utc_now=lambda: fake.now)
    assert other.get_float("gdhg").float_shares == 1_000_000 and fake.sent == ["GDHG"]
    fake.now += timedelta(days=6.9)
    client.get_float("GDHG")
    assert fake.sent == ["GDHG"]
    fake.now += timedelta(days=0.2)
    client.get_float("GDHG")
    assert fake.sent == ["GDHG", "GDHG"]


def test_not_found_kept_until_the_next_eastern_day(setup):
    client, fake, _ = setup
    fake.answers["ZQZQX"] = (200, NOT_FOUND)
    assert client.get_float("ZQZQX") is None
    assert client.get_float("ZQZQX") is None and fake.sent == ["ZQZQX"]
    fake.now += timedelta(days=1)
    client.get_float("ZQZQX")
    assert fake.sent == ["ZQZQX", "ZQZQX"]


def test_doubtful_float_not_used(setup):
    client, fake, _ = setup
    fake.answers["FGL"] = (200, _stats("FGL", "8,557", "11,456", "1,565,709"))
    assert client.get_float("FGL") is None
    assert client.facts("FGL")["float"] == 8_557  # still cached, as ScalpFeed shows it with a ?


def test_share_classes_asked_with_a_dot(setup):
    client, fake, _ = setup
    client.get_float("BRK/B")
    assert fake.sent == ["BRK.B"]


def test_failures_wait_and_out_of_credits_pauses_everything(setup):
    client, fake, _ = setup
    fake.answers["AAA"] = (500, "")
    assert client.get_float("AAA") is None
    assert client.get_float("AAA") is None and fake.sent == ["AAA"]  # retried after 10 minutes, not now
    fake.answers["BBB"] = (402, "")
    assert client.get_float("BBB") is None
    assert client.get_float("CCC") is None and fake.sent == ["AAA", "BBB"]


def test_writes_merge_with_scalpfeeds_entries(setup):
    client, fake, tmp_path = setup
    path = str(tmp_path / "float-cache.json")
    # Written by ScalpFeed (FloatFacts.WriteCache): nulls for unknowns, mixed-case keys read as one.
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"GMM": {"day": "2026-11-16", "float": 2_500_000, "sharesOut": None, "averageVolume": None,
                           "fetchedMs": int(fake.now.timestamp() * 1000)}}, handle)
    client.get_float("GDHG")
    cache = ff.read_cache(path)
    assert set(cache) == {"GMM", "GDHG"}
    assert cache["GMM"]["float"] == 2_500_000 and math.isnan(cache["GMM"]["sharesOut"])
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    assert raw["GMM"]["sharesOut"] is None  # NaN goes back out as null, which ScalpFeed reads


def test_eastern_day_boundary():
    assert ff.eastern_day(datetime(2026, 11, 16, 3, 0, tzinfo=timezone.utc)) == "2026-11-15"
    assert ff.eastern_day(datetime(2026, 11, 16, 9, 0, tzinfo=timezone.utc)) == "2026-11-16"
