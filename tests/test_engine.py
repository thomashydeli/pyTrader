import pandas as pd
import pytest

from algorithm.base import Algorithm
from backtest.engine import Backtester


class FixedWeights(Algorithm):
    """Test double: ignores prices, always returns the weights it was built with."""

    def __init__(self, weights, columns):
        self.weights, self.columns = weights, columns

    def generate_target_weights(self, prices):
        return pd.DataFrame(self.weights, index=prices.index, columns=self.columns)


@pytest.fixture
def toy_prices():
    idx = pd.bdate_range("2020-01-01", periods=5)
    # returns: -, +10%, +10%, 0%, -10%
    return pd.DataFrame({"X": [100, 110, 121, 121, 108.9]}, index=idx)


def test_equity_curve_matches_hand_calculation(toy_prices):
    # weights: [0, 1, 1, 0, 0] -> shift(1): [0, 0, 1, 1, 0]
    # gross returns: [0, 0, 0.10, 0.00, 0.00]
    # turnover (diff, day0 = |w0|): [0, 1, 0, 1, 0]; cost @ 100bps: [0, .01, 0, .01, 0]
    # net returns:  [0, -0.01, 0.10, -0.01, 0]
    # equity (start 1000): 1000, 990, 1089, 1078.11, 1078.11
    algo = FixedWeights({"X": [0, 1, 1, 0, 0]}, columns=["X"])
    result = Backtester().run(algo, toy_prices, initial_capital=1000.0, commission_bps=100.0)

    assert result.returns.tolist() == pytest.approx([0.0, -0.01, 0.10, -0.01, 0.0])
    assert result.equity.tolist() == pytest.approx([1000.0, 990.0, 1089.0, 1078.11, 1078.11])


def test_no_look_ahead_bias_first_day_return_is_always_zero(toy_prices):
    # even a strategy that (unrealistically) wants full exposure on day 0
    # cannot earn day-0's return, since there was no prior-day position.
    algo = FixedWeights({"X": [1, 1, 1, 1, 1]}, columns=["X"])
    result = Backtester().run(algo, toy_prices)
    assert result.returns.iloc[0] == 0.0


def test_zero_cost_is_deterministic_and_reproducible(toy_prices):
    algo = FixedWeights({"X": [0, 1, 1, 0, 0]}, columns=["X"])
    result_a = Backtester().run(algo, toy_prices)
    result_b = Backtester().run(algo, toy_prices)
    assert result_a.equity.tolist() == result_b.equity.tolist()


def test_raises_on_leverage_violation(toy_prices):
    algo = FixedWeights({"X": [0, 1.5, 1.5, 0, 0]}, columns=["X"])
    with pytest.raises(ValueError, match="leverage cap"):
        Backtester().run(algo, toy_prices)


def test_raises_on_nan_weights(toy_prices):
    algo = FixedWeights({"X": [0, float("nan"), 1, 0, 0]}, columns=["X"])
    with pytest.raises(ValueError, match="NaN"):
        Backtester().run(algo, toy_prices)


def test_raises_on_unknown_ticker_in_weights(toy_prices):
    # a strategy bug that names a ticker absent from `prices` must fail loudly,
    # not silently get dropped by reindexing.
    algo = FixedWeights({"X": [0, 1, 1, 0, 0], "Y": [0, 0, 0, 0, 0]}, columns=["X", "Y"])
    with pytest.raises(ValueError, match="Y"):
        Backtester().run(algo, toy_prices)


def test_extreme_short_squeeze_floors_at_zero_instead_of_going_negative():
    # a full short position (-1.0) the day before a >100% single-day price
    # spike would otherwise drive net_returns below -100% and flip equity's
    # sign for every subsequent day -- it must floor at total loss (0) instead.
    idx = pd.bdate_range("2020-01-01", periods=2)
    prices = pd.DataFrame({"X": [100.0, 250.0]}, index=idx)  # +150% in one day
    algo = FixedWeights({"X": [-1.0, -1.0]}, columns=["X"])

    result = Backtester().run(algo, prices, initial_capital=1000.0)

    assert result.returns.tolist() == [0.0, -1.0]
    assert result.equity.tolist() == [1000.0, 0.0]


def test_raises_when_weights_index_does_not_match_prices(toy_prices):
    class BadIndex(Algorithm):
        def generate_target_weights(self, prices):
            return pd.DataFrame({"X": [0, 1, 1, 0]}, index=prices.index[:4])

    with pytest.raises(ValueError, match="index"):
        Backtester().run(BadIndex(), toy_prices)
