import pandas as pd

from algorithm import PairsTrading

# B is a fixed random walk; A tracks it with small idiosyncratic noise, except
# for a deliberate 5-day dislocation (rows 25-29) well beyond the noise floor.
# (Generated once with numpy seed=3, then hardcoded so the test data itself
# never changes regardless of implementation details.)
B = [
    101.7886, 102.2251, 102.3216, 100.4581, 100.1808, 99.8260, 99.7433, 99.1163, 99.0724, 98.5952,
    97.2814, 98.1660, 99.0473, 100.7569, 100.8069, 100.4022, 99.8569, 98.3104, 99.2928, 98.1917,
    97.0066, 96.8010, 98.2871, 98.5239, 97.5001, 96.7871, 97.4123, 97.2518, 96.4830, 96.2529,
    96.9980, 98.9741, 97.7300, 97.1036, 96.2998, 93.8807, 92.9569, 91.9330, 93.0570, 92.9251,
]
A = [
    101.3016, 102.4191, 102.2148, 99.9352, 100.0018, 99.6494, 99.4811, 99.1252, 98.3980, 98.5149,
    97.5853, 98.4218, 99.3797, 101.0927, 101.2532, 100.0667, 100.1106, 97.7521, 99.1119, 97.6173,
    97.3211, 97.2011, 98.2279, 99.0562, 97.2976, 111.8323, 112.4582, 111.9325, 111.6144, 111.8346,
    96.6905, 99.2439, 97.6836, 97.6345, 96.4449, 94.0836, 93.1499, 92.0078, 92.6383, 93.3426,
]


def _prices():
    idx = pd.bdate_range("2020-01-01", periods=len(B))
    return pd.DataFrame({"A": pd.Series(A, index=idx), "B": pd.Series(B, index=idx)})


def test_warm_up_period_is_zero_not_nan():
    prices = _prices()
    algo = PairsTrading("A", "B", window=10, entry_z=2.0, exit_z=0.5)
    weights = algo.generate_target_weights(prices)
    assert weights.isna().sum().sum() == 0
    # the z-score needs two full rolling windows (hedge ratio, then its own
    # rolling mean/std) before it's defined for the first time
    assert (weights.iloc[:18] == 0.0).all().all()


def test_entry_exit_sequence():
    # The z-score crosses +entry_z at row 24 (dislocation: short A/long B),
    # holds through the noise, then crosses -entry_z at row 35 once the
    # dislocation unwinds (flips to long A/short B) -- verified by direct
    # computation, with a wide margin from both thresholds at every row.
    prices = _prices()
    algo = PairsTrading("A", "B", window=10, entry_z=2.0, exit_z=0.5)
    weights = algo.generate_target_weights(prices)

    expected_a = [0.0] * 24 + [-0.5] * 11 + [0.5] * 5
    assert weights["A"].tolist() == expected_a
    assert weights["B"].tolist() == [-w for w in expected_a]


def test_legs_are_mirrored_and_within_leverage_cap():
    prices = _prices()
    weights = PairsTrading("A", "B", window=10, entry_z=2.0, exit_z=0.5).generate_target_weights(prices)
    assert (weights["A"] == -weights["B"]).all()
    assert (weights.abs().sum(axis=1) <= 1.0 + 1e-9).all()
