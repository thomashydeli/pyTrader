"""Adjusted-close price loading with a local parquet cache.

The rest of the library only ever calls :func:`get_prices`, so the data
source (currently yfinance) can be swapped later without touching anything
downstream.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache"


def get_prices(tickers, start, end, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Return a clean (no gaps, no NaNs) DataFrame of adjusted close prices.

    Rows are trading dates in ``[start, end]``; columns are ``tickers``.
    Raises loudly instead of silently dropping/filling bad data.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(dict.fromkeys(tickers))  # de-dupe, preserving order -- a duplicate
    # ticker would otherwise produce a repeated column name that breaks `prices[ticker]`
    # lookups downstream (returns a DataFrame instead of a Series).
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    prices = pd.concat(
        [_load_ticker(t, start, end, cache_dir) for t in tickers], axis=1
    ).sort_index()
    prices = prices.loc[(prices.index >= start) & (prices.index <= end)]

    gaps = prices.isna().sum()
    gaps = gaps[gaps > 0]
    if not gaps.empty:
        raise ValueError(
            f"Excessive gaps in price data, missing trading days per ticker: "
            f"{gaps.to_dict()}. Check the ticker symbols or narrow the date range."
        )
    return prices


def _load_ticker(ticker: str, start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path) -> pd.Series:
    """Fetch and cache one ticker, remembering the *requested* range already
    fetched (not just the trading days found in it) so a holiday/weekend at a
    range edge is never re-probed on every call."""
    data_path = cache_dir / f"{ticker}.parquet"
    range_path = cache_dir / f"{ticker}.range"
    cached = _read_cache(data_path)
    fetched_start, fetched_end = _read_range(range_path)
    if cached is not None and fetched_start is None:  # no readable range info -- trust the data we have
        fetched_start, fetched_end = cached.index.min(), cached.index.max()

    if cached is None:
        merged = _download(ticker, start, end)
        if merged.empty:
            raise RuntimeError(f"Failed to download any data for ticker '{ticker}'.")
        fetched_start, fetched_end = start, end
    else:
        merged = cached
        if start < fetched_start:
            merged = pd.concat([_download(ticker, start, fetched_start - pd.Timedelta(days=1)), merged])
            fetched_start = start
        if end > fetched_end:
            merged = pd.concat([merged, _download(ticker, fetched_end + pd.Timedelta(days=1), end)])
            fetched_end = end
        merged = merged.sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]

    if cached is None or not merged.equals(cached):
        _write_cache(data_path, merged)
    _write_range(range_path, fetched_start, fetched_end)

    result = merged.loc[start:end]
    if result.empty:
        raise RuntimeError(f"No data available for '{ticker}' between {start.date()} and {end.date()}.")
    return result.rename(ticker)


def _read_cache(path: Path):
    if not path.exists():
        return None
    return pd.read_parquet(path).iloc[:, 0]


def _write_cache(path: Path, series: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame(name="adj_close").to_parquet(path)


def _read_range(path: Path):
    """(None, None) if the sidecar is missing, truncated, or otherwise unreadable --
    treated the same as "unknown", not as an error worth failing the whole call over."""
    try:
        start, end = path.read_text().split(",")
        return pd.Timestamp(start), pd.Timestamp(end)
    except (OSError, ValueError):
        return None, None


def _write_range(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{start.isoformat()},{end.isoformat()}")


def _download(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if start > end:
        return pd.Series(dtype=float)
    df = yf.download(
        ticker, start=start, end=end + pd.Timedelta(days=1), auto_adjust=True, progress=False
    )
    if df.empty:
        return pd.Series(dtype=float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance may return MultiIndex columns
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.astype(float).rename(ticker)
