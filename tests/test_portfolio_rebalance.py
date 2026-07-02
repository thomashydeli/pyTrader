import pandas as pd

from algorithm import PortfolioRebalance


def test_weights_held_flat_at_target_across_a_month_boundary():
    idx = pd.bdate_range("2020-01-28", periods=8)  # spans the Jan -> Feb boundary
    prices = pd.DataFrame({"A": range(100, 108), "B": range(50, 58)}, index=idx, dtype=float)

    weights = PortfolioRebalance({"A": 0.6, "B": 0.4}).generate_target_weights(prices)

    assert weights.index.equals(prices.index)
    assert weights.isna().sum().sum() == 0
    assert (weights["A"] == 0.6).all()
    assert (weights["B"] == 0.4).all()


def test_invested_from_the_first_trading_day():
    idx = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [10, 11, 12]}, index=idx, dtype=float)

    weights = PortfolioRebalance({"A": 0.5}).generate_target_weights(prices)

    assert weights["A"].iloc[0] == 0.5


def test_respects_leverage_cap():
    idx = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [10, 11, 12], "B": [20, 21, 22]}, index=idx, dtype=float)

    weights = PortfolioRebalance({"A": 0.6, "B": 0.4}).generate_target_weights(prices)

    assert (weights.abs().sum(axis=1) <= 1.0 + 1e-9).all()
