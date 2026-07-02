from abc import ABC, abstractmethod

import pandas as pd


class Algorithm(ABC):
    """Every strategy decides *what to hold*; the backtester turns that into an equity curve."""

    @abstractmethod
    def generate_target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return target portfolio-equity fractions per ticker, indexed like ``prices``.

        Values must be in [-1, 1] with ``sum(abs(weights)) <= 1`` on every date, and
        ``0`` (not ``NaN``) before the strategy has enough history to signal.
        """
