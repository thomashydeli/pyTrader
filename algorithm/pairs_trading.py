import pandas as pd

from .base import Algorithm


class PairsTrading(Algorithm):
    """Two stocks: trade the spread's z-score using a rolling OLS hedge ratio.

    Long-A/short-B below ``-entry_z``, short-A/long-B above ``+entry_z``, flatten
    once the z-score returns inside ``exit_z`` (hysteresis, not a same-bar flip-flop).
    """

    def __init__(self, ticker_a: str, ticker_b: str, window: int = 60, entry_z: float = 2.0, exit_z: float = 0.5):
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def compute_spread_zscore(self, prices: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Spread and rolling z-score, exposed for charting (see ``report.pairs_spread_chart``)."""
        a, b = prices[self.ticker_a], prices[self.ticker_b]
        hedge_ratio = _rolling_hedge_ratio(a, b, self.window)
        spread = a - hedge_ratio * b
        z = (spread - spread.rolling(self.window).mean()) / spread.rolling(self.window).std()
        return spread, z

    def generate_target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        _, z = self.compute_spread_zscore(prices)

        long_a = (z < -self.entry_z).fillna(False)
        short_a = (z > self.entry_z).fillna(False)
        flat = (z.abs() < self.exit_z).fillna(False)

        position = 0
        states = []
        for lo, sh, fl in zip(long_a, short_a, flat):
            if fl:
                position = 0
            elif lo:
                position = 1
            elif sh:
                position = -1
            states.append(position)

        side = pd.Series(states, index=prices.index)
        weights = pd.DataFrame(0.0, index=prices.index, columns=[self.ticker_a, self.ticker_b])
        weights[self.ticker_a] = side * 0.5
        weights[self.ticker_b] = -side * 0.5
        return weights


def _rolling_hedge_ratio(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope of A on B (A = ratio * B + intercept): cov(A, B) / var(B)."""
    return b.rolling(window).cov(a) / b.rolling(window).var()
