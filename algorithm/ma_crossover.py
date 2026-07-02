import pandas as pd

from .base import Algorithm


class MovingAverageCrossover(Algorithm):
    """Single stock: long when short MA > long MA, else flat (or short, if enabled)."""

    def __init__(self, ticker: str, short_window: int = 20, long_window: int = 100, allow_short: bool = False):
        self.ticker = ticker
        self.short_window = short_window
        self.long_window = long_window
        self.allow_short = allow_short

    def generate_target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        price = prices[self.ticker]
        short_ma = price.rolling(self.short_window).mean()
        long_ma = price.rolling(self.long_window).mean()
        warm_up = long_ma.isna()

        weight = pd.Series(0.0, index=prices.index)
        weight[~warm_up & (short_ma > long_ma)] = 1.0
        if self.allow_short:
            weight[~warm_up & (short_ma <= long_ma)] = -1.0
        return weight.to_frame(self.ticker)
