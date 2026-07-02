import pandas as pd
import pytest

from algorithm import MovingAverageCrossover


@pytest.fixture
def prices():
    idx = pd.bdate_range("2020-01-01", periods=6)
    # short_window=2, long_window=3 -> long_ma warms up at index 2.
    # short_ma: [-, 10.5, 11.5, 10.5, 8.5, 11.5]
    # long_ma:  [-, -,    11.0, 10.667, 9.667, 10.667]
    return pd.DataFrame({"X": [10, 11, 12, 9, 8, 15]}, index=idx)


def test_warm_up_period_is_zero_not_nan(prices):
    weights = MovingAverageCrossover("X", short_window=2, long_window=3).generate_target_weights(prices)
    assert weights["X"].isna().sum() == 0
    assert weights["X"].iloc[:2].tolist() == [0.0, 0.0]


def test_long_only_signal_matches_hand_calculation(prices):
    weights = MovingAverageCrossover("X", short_window=2, long_window=3).generate_target_weights(prices)
    assert weights["X"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]


def test_allow_short_flips_flat_to_short(prices):
    weights = MovingAverageCrossover("X", short_window=2, long_window=3, allow_short=True).generate_target_weights(prices)
    assert weights["X"].tolist() == [0.0, 0.0, 1.0, -1.0, -1.0, 1.0]


def test_output_shares_input_index_and_leverage_cap(prices):
    weights = MovingAverageCrossover("X", short_window=2, long_window=3).generate_target_weights(prices)
    assert weights.index.equals(prices.index)
    assert (weights.abs().sum(axis=1) <= 1.0 + 1e-9).all()
