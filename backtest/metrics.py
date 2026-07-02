import numpy as np
import pandas as pd

from .engine import TRADING_DAYS_PER_YEAR, BacktestResult, turnover as _turnover


def total_return(equity: pd.Series) -> float:
    return equity.iloc[-1] / equity.iloc[0] - 1


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    years = len(equity) / periods_per_year
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0


def annual_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    excess = returns - risk_free / periods_per_year
    vol = returns.std()
    # `if vol` is wrong here: NaN (e.g. a 1-row series) is truthy in Python,
    # so only the exact-zero case would be caught. `vol > 0` is False for
    # both 0 and NaN, which is what we want.
    return float(excess.mean() / vol * np.sqrt(periods_per_year)) if vol > 0 else 0.0


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    excess = returns - risk_free / periods_per_year
    downside = returns[returns < 0].std()
    # A strategy with zero losing days makes `downside` NaN (std of an empty
    # series), not 0 -- same NaN-is-truthy pitfall as sharpe_ratio above.
    return float(excess.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def calmar_ratio(cagr_value: float, max_dd: float) -> float:
    return cagr_value / abs(max_dd) if max_dd < 0 else 0.0


def win_rate(returns: pd.Series) -> float:
    traded = returns[returns != 0]
    return float((traded > 0).mean()) if len(traded) else 0.0


def trade_count(weights: pd.DataFrame) -> int:
    return int((_turnover(weights) > 1e-9).sum())


def compute_metrics(result: BacktestResult, benchmark_prices: pd.Series | None = None) -> dict:
    equity, returns = result.equity, result.returns
    dd = max_drawdown(equity)
    stats = {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "annual_volatility": annual_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": dd,
        "calmar": calmar_ratio(cagr(equity), dd),
        "win_rate": win_rate(returns),
        "trade_count": trade_count(result.weights),
    }
    if benchmark_prices is not None:
        bench_equity = (1 + benchmark_prices.pct_change().reindex(equity.index).fillna(0.0)).cumprod()
        stats["benchmark_total_return"] = total_return(bench_equity)
        stats["benchmark_cagr"] = cagr(bench_equity)
    return stats
