from dataclasses import dataclass

import pandas as pd

from algorithm.base import Algorithm

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    prices: pd.DataFrame
    weights: pd.DataFrame
    returns: pd.Series  # net daily returns
    equity: pd.Series
    initial_capital: float
    commission_bps: float
    slippage_bps: float


class Backtester:
    """Vectorized, deterministic daily-bar simulator (not an order/fill simulator)."""

    def run(
        self,
        algorithm: Algorithm,
        prices: pd.DataFrame,
        initial_capital: float = 100_000.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
    ) -> BacktestResult:
        weights = algorithm.generate_target_weights(prices)
        validate_weights(weights, prices)
        weights = weights.reindex(columns=prices.columns, fill_value=0.0)

        gross_returns = (weights.shift(1).fillna(0.0) * prices.pct_change()).sum(axis=1)
        net_returns = gross_returns - turnover(weights) * (commission_bps + slippage_bps) / 1e4
        # A single day can't lose more than the capital allocated to it: this floor
        # only bites on an extreme short-side blowup (e.g. a >100% single-day squeeze),
        # where compounding through negative equity would otherwise flip its sign
        # every subsequent day instead of just going to (and staying at) zero.
        net_returns = net_returns.clip(lower=-1.0)
        equity = initial_capital * (1 + net_returns).cumprod()

        return BacktestResult(
            prices=prices,
            weights=weights,
            returns=net_returns,
            equity=equity,
            initial_capital=initial_capital,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Day-over-day change in gross exposure; day 0 counts the cost of the initial position."""
    daily_turnover = weights.diff().abs().sum(axis=1)
    daily_turnover.iloc[0] = weights.iloc[0].abs().sum()
    return daily_turnover


def validate_weights(weights: pd.DataFrame, prices: pd.DataFrame) -> None:
    if not weights.index.equals(prices.index):
        raise ValueError("Algorithm output weights must share the exact date index of `prices`.")
    unknown = set(weights.columns) - set(prices.columns)
    if unknown:
        raise ValueError(f"Algorithm output weights for tickers not present in `prices`: {sorted(unknown)}.")
    if weights.isna().any().any():
        raise ValueError("Algorithm output weights contain NaNs; strategies must emit 0.0 during warm-up.")
    leverage = weights.abs().sum(axis=1)
    breaches = leverage[leverage > 1 + 1e-9]
    if not breaches.empty:
        raise ValueError(
            f"Weights violate the leverage cap (sum(abs(weights)) <= 1) on "
            f"{len(breaches)} date(s); max total exposure was {leverage.max():.4f}."
        )
