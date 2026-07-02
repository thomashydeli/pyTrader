import pandas as pd
import pytest

from backtest import metrics


@pytest.fixture
def equity():
    # +10%, +10%, -10%, 0% -> peak 121, trough 108.9 -> drawdown exactly -10%
    return pd.Series([100.0, 110.0, 121.0, 108.9, 108.9])


def test_total_return(equity):
    assert metrics.total_return(equity) == pytest.approx(0.089)


def test_max_drawdown(equity):
    assert metrics.max_drawdown(equity) == pytest.approx(-0.10)


def test_win_rate_ignores_zero_return_days():
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, -0.01])
    # 2 of the 4 nonzero days are positive
    assert metrics.win_rate(returns) == pytest.approx(0.5)


def test_trade_count_counts_weight_changes():
    weights = pd.DataFrame({"A": [0.0, 1.0, 1.0, 0.0, 0.5]})
    assert metrics.trade_count(weights) == 3


def test_trade_count_includes_day_zero_entry():
    # matches the engine's own turnover accounting: being invested from day 0
    # is itself a trade, mirroring Backtester's `turnover.iloc[0]` cost.
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0]})
    assert metrics.trade_count(weights) == 1


def test_sharpe_ratio_zero_when_no_volatility():
    assert metrics.sharpe_ratio(pd.Series([0.0, 0.0, 0.0])) == 0.0


def test_sharpe_ratio_zero_when_std_is_nan_not_just_zero():
    # a single-row returns series makes .std() NaN, not 0 -- `if vol` would
    # treat NaN as truthy in Python and wrongly skip the zero-vol fallback.
    assert metrics.sharpe_ratio(pd.Series([0.01])) == 0.0


def test_sortino_ratio_zero_when_no_losing_days():
    # zero negative-return days makes the downside deviation NaN, not 0 --
    # the most common way this NaN-is-truthy pitfall would actually be hit.
    assert metrics.sortino_ratio(pd.Series([0.01, 0.02, 0.0, 0.015])) == 0.0


def test_calmar_ratio_zero_when_no_drawdown():
    assert metrics.calmar_ratio(cagr_value=0.10, max_dd=0.0) == 0.0


def test_calmar_ratio_zero_when_drawdown_is_nan():
    assert metrics.calmar_ratio(cagr_value=0.10, max_dd=float("nan")) == 0.0
