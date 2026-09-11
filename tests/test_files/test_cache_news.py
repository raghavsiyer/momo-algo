from datetime import datetime, timedelta, timezone
from io import BytesIO
from types import SimpleNamespace

from cache import TickerCache, _CacheUnpickler, apply_cached_news
from market_data.finnhub_client import FinnhubNewsArticle
from strategy.evaluator import CompositeEvaluation


def _article(title: str, published_at: datetime) -> FinnhubNewsArticle:
    return FinnhubNewsArticle(
        symbol="AZI",
        title=title,
        source="momo",
        url="",
        published_at=published_at,
        summary="",
    )


class _Evaluation:
    def __init__(self) -> None:
        self.updated = None
        self.stock_eval = SimpleNamespace(score=0.86)

    def update_news(self, avg_sentiment, news_count, news_source):
        self.updated = (avg_sentiment, news_count, news_source)


def test_news_cache_persists_between_instances(tmp_path, monkeypatch):
    path = tmp_path / "ticker_cache.pkl"
    article = _article("AZI announces catalyst", datetime.now(tz=timezone.utc))
    cache = TickerCache(path=path)
    monkeypatch.setattr(cache, "_maybe_snapshot", lambda: None)

    cache.put_news("AZI", [article], 1, 0.7, "momo")
    cache.flush()

    reloaded = TickerCache(path=path)
    cached_news = reloaded.get_news("AZI")

    assert cached_news is not None
    articles, count, sentiment, source = cached_news
    assert articles[0].title == "AZI announces catalyst"
    assert count == 1
    assert sentiment == 0.7
    assert source == "momo"


def test_news_cache_suppresses_stale_articles(tmp_path, monkeypatch):
    cache = TickerCache(path=tmp_path / "ticker_cache.pkl")
    monkeypatch.setattr(cache, "_maybe_snapshot", lambda: None)
    old_article = _article("AZI old catalyst", datetime.now(tz=timezone.utc) - timedelta(days=10))

    cache.put_news("AZI", [old_article], 1, 0.4, "momo")

    assert cache.get_news("AZI") is None


def test_apply_cached_news_updates_entry_and_evaluation():
    article = _article("AZI fresh catalyst", datetime.now(tz=timezone.utc))
    evaluation = _Evaluation()
    entry = SimpleNamespace(
        news_articles=[],
        news_count_last_hour=0,
        avg_sentiment=None,
        news_source="",
        news_is_fresh=False,
        news_is_breaking=False,
        evaluation=evaluation,
        static_confidence=0.0,
    )

    updated = apply_cached_news(entry, ([article], 2, 0.8, "momo"))

    assert updated is True
    assert entry.news_articles[0].title == "AZI fresh catalyst"
    assert entry.news_count_last_hour == 2
    assert entry.avg_sentiment == 0.8
    assert entry.news_source == "momo"
    assert entry.static_confidence == 86.0
    assert evaluation.updated == (0.8, 2, "momo")


def test_cache_unpickler_maps_legacy_evaluation_class():
    unpickler = _CacheUnpickler(BytesIO())

    resolved = unpickler.find_class("strategy.evaluator", "StaticEvaluation")

    assert resolved is CompositeEvaluation