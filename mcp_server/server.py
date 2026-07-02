"""Thin MCP tool wrappers over the core library -- no business logic lives here.

Read-only + simulation only: nothing here can place an order. A future live
broker adapter would be driven by ``get_current_signal``'s output, from outside
this module, and must require explicit opt-in.
"""
import uuid
from datetime import date, timedelta

from mcp.server.fastmcp import FastMCP

from algorithm import MovingAverageCrossover, PairsTrading, PortfolioRebalance
from backtest import Backtester, Report, build_report, validate_weights
from data import get_prices

ALGORITHMS = {
    "moving_average_crossover": MovingAverageCrossover,
    "pairs_trading": PairsTrading,
    "portfolio_rebalance": PortfolioRebalance,
}

PARAM_SCHEMAS = {
    "moving_average_crossover": {
        "ticker": "str", "short_window": "int = 20", "long_window": "int = 100", "allow_short": "bool = False",
    },
    "pairs_trading": {
        "ticker_a": "str", "ticker_b": "str", "window": "int = 60", "entry_z": "float = 2.0", "exit_z": "float = 0.5",
    },
    "portfolio_rebalance": {"target_weights": "dict[str, float]", "frequency": "str = 'ME'"},
}

mcp = FastMCP("pytrader")
_reports: dict[str, Report] = {}


@mcp.tool()
def list_algorithms() -> dict:
    """List built-in algorithm names and their constructor parameter schema."""
    return PARAM_SCHEMAS


@mcp.tool()
def run_backtest(
    algorithm: str,
    params: dict,
    tickers: list[str],
    start: str,
    end: str,
    initial_capital: float = 100_000.0,
    commission_bps: float = 0.0,
) -> dict:
    """Run a backtest and return a backtest_id plus summary stats."""
    prices = get_prices(tickers, start, end)
    instance = ALGORITHMS[algorithm](**params)
    result = Backtester().run(instance, prices, initial_capital=initial_capital, commission_bps=commission_bps)
    report = build_report(result)

    backtest_id = str(uuid.uuid4())
    _reports[backtest_id] = report
    return {"backtest_id": backtest_id, "stats": report.stats}


@mcp.tool()
def get_backtest_report(backtest_id: str) -> dict:
    """Stats JSON + chart JSON (Plotly figures) for a prior run."""
    report = _reports.get(backtest_id)
    if report is None:
        raise ValueError(f"Unknown backtest_id: {backtest_id!r}")
    return {"stats": report.stats, "figures": [fig.to_json() for fig in report.figures]}


@mcp.tool()
def get_current_signal(algorithm: str, params: dict, tickers: list[str]) -> dict:
    """Today's target weights only, no simulation -- the seam a live broker would call."""
    end = date.today()
    start = end - timedelta(days=400)  # enough history to warm up any built-in indicator
    prices = get_prices(tickers, start, end)
    instance = ALGORITHMS[algorithm](**params)
    weights = instance.generate_target_weights(prices)
    validate_weights(weights, prices)  # this is the seam a live broker would trust -- never skip it
    return weights.iloc[-1].to_dict()


if __name__ == "__main__":
    mcp.run()
