import pandas as pd
import pytest

import data.loader as loader


def _fake_series(ticker, start, end):
    idx = pd.bdate_range(start, end)
    return pd.Series(range(len(idx)), index=idx, dtype=float, name=ticker)


def test_get_prices_caches_and_avoids_redundant_downloads(monkeypatch, tmp_path):
    calls = []

    def fake_download(ticker, start, end):
        calls.append((start, end))
        return _fake_series(ticker, start, end)

    monkeypatch.setattr(loader, "_download", fake_download)

    prices = loader.get_prices(["AAPL"], "2020-01-06", "2020-01-10", cache_dir=tmp_path)
    assert not prices.isna().any().any()
    assert len(calls) == 1

    # fully covered by the cache -> no new download
    loader.get_prices(["AAPL"], "2020-01-07", "2020-01-08", cache_dir=tmp_path)
    assert len(calls) == 1

    # only the missing earlier days should be re-fetched, not the cached range
    loader.get_prices(["AAPL"], "2020-01-01", "2020-01-10", cache_dir=tmp_path)
    assert len(calls) == 2
    assert calls[1][1] < calls[0][0]  # second fetch ends before the first fetch's start


def test_holiday_at_range_edge_is_not_reprobed_on_every_call(monkeypatch, tmp_path):
    # 2020-01-01 is a Wednesday with no trading data (simulating a market holiday).
    # A naive cache that compares against the *data* it found (not what it asked
    # for) would retry fetching that empty day forever.
    calls = []

    def fake_download(ticker, start, end):
        calls.append((start, end))
        series = _fake_series(ticker, start, end)
        return series.drop(pd.Timestamp("2020-01-01"), errors="ignore")

    monkeypatch.setattr(loader, "_download", fake_download)

    loader.get_prices(["AAPL"], "2020-01-01", "2020-01-10", cache_dir=tmp_path)
    assert len(calls) == 1

    loader.get_prices(["AAPL"], "2020-01-01", "2020-01-10", cache_dir=tmp_path)
    loader.get_prices(["AAPL"], "2020-01-01", "2020-01-10", cache_dir=tmp_path)
    assert len(calls) == 1  # never re-probed the known-empty holiday


def test_tolerates_cache_predating_the_range_sidecar_file(monkeypatch, tmp_path):
    # a .parquet file with no matching .range file (e.g. written by an older
    # version of this loader, or dropped in by hand) must not crash.
    cached = _fake_series("AAPL", "2020-01-06", "2020-01-10")
    cached.to_frame(name="adj_close").to_parquet(tmp_path / "AAPL.parquet")
    assert not (tmp_path / "AAPL.range").exists()

    monkeypatch.setattr(loader, "_download", lambda ticker, start, end: _fake_series(ticker, start, end))
    prices = loader.get_prices(["AAPL"], "2020-01-07", "2020-01-08", cache_dir=tmp_path)
    assert not prices.isna().any().any()


def test_tolerates_a_malformed_range_sidecar_file(monkeypatch, tmp_path):
    cached = _fake_series("AAPL", "2020-01-06", "2020-01-10")
    cached.to_frame(name="adj_close").to_parquet(tmp_path / "AAPL.parquet")
    (tmp_path / "AAPL.range").write_text("not,a,valid,range,file")

    monkeypatch.setattr(loader, "_download", lambda ticker, start, end: _fake_series(ticker, start, end))
    prices = loader.get_prices(["AAPL"], "2020-01-07", "2020-01-08", cache_dir=tmp_path)
    assert not prices.isna().any().any()


def test_duplicate_tickers_are_deduped_not_left_to_crash_downstream(monkeypatch, tmp_path):
    # a duplicate ticker would otherwise produce a repeated column name, and
    # `prices[ticker]` inside an Algorithm returns a DataFrame instead of a
    # Series when that happens.
    monkeypatch.setattr(loader, "_download", lambda ticker, start, end: _fake_series(ticker, start, end))
    prices = loader.get_prices(["AAPL", "AAPL"], "2020-01-06", "2020-01-10", cache_dir=tmp_path)
    assert list(prices.columns) == ["AAPL"]


def test_fails_loudly_when_ticker_has_no_data(monkeypatch, tmp_path):
    monkeypatch.setattr(loader, "_download", lambda ticker, start, end: pd.Series(dtype=float))
    with pytest.raises(RuntimeError, match="BADTICKER"):
        loader.get_prices(["BADTICKER"], "2020-01-01", "2020-01-10", cache_dir=tmp_path)


def test_fails_loudly_on_gaps_between_tickers(monkeypatch, tmp_path):
    def fake_download(ticker, start, end):
        series = _fake_series(ticker, start, end)
        return series.drop(series.index[2]) if ticker == "GAPPY" else series

    monkeypatch.setattr(loader, "_download", fake_download)

    with pytest.raises(ValueError, match="gaps"):
        loader.get_prices(["AAPL", "GAPPY"], "2020-01-06", "2020-01-10", cache_dir=tmp_path)
