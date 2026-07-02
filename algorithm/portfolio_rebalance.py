import pandas as pd

from .base import Algorithm


class PortfolioRebalance(Algorithm):
    """N stocks held at fixed target weights, re-applied on a calendar schedule.

    Weights are held flat (ffill) between rebalance dates, since the engine's
    ``weights.shift(1) * returns`` already carries yesterday's target forward
    with zero turnover until the next scheduled reset.
    """

    def __init__(self, target_weights: dict, frequency: str = "ME"):
        self.target_weights = target_weights
        self.frequency = frequency  # pandas offset alias, e.g. "ME" (month), "QE" (quarter), "W"

    def generate_target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        tickers = list(self.target_weights.keys())
        target_row = pd.Series(self.target_weights, index=tickers)

        first_of_period = pd.Series(prices.index, index=prices.index).resample(self.frequency).first().dropna()
        rebalance_dates = pd.DatetimeIndex(first_of_period.values).union([prices.index[0]])

        weights = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
        weights.loc[weights.index.isin(rebalance_dates)] = target_row.values
        return weights.ffill().fillna(0.0)
