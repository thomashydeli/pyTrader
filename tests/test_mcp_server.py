import pandas as pd
import pytest

import mcp_server.server as server
from algorithm.base import Algorithm


class BadAlgorithm(Algorithm):
    """Test double that violates the weight contract, to prove get_current_signal
    actually enforces it rather than handing bad data straight to a caller that
    (per this module's own docstring) may be a live broker adapter."""

    def generate_target_weights(self, prices):
        return pd.DataFrame({"X": [2.0] * len(prices)}, index=prices.index)  # exceeds leverage cap


def test_get_current_signal_validates_weights_before_returning(monkeypatch):
    monkeypatch.setattr(server, "ALGORITHMS", {**server.ALGORITHMS, "bad": BadAlgorithm})
    monkeypatch.setattr(server, "get_prices", lambda tickers, start, end: pd.DataFrame(
        {"X": [100.0, 101.0, 102.0]}, index=pd.bdate_range("2020-01-01", periods=3)
    ))

    with pytest.raises(ValueError, match="leverage cap"):
        server.get_current_signal("bad", {}, ["X"])
