# pyTrader

A lightweight backtesting toolkit for US equities. One interface — `Algorithm.generate_target_weights(prices) -> weights` — is shared by every built-in strategy, your own strategies, the interactive dashboard, and the MCP server. See [doc/design.md](doc/design.md) for the full design rationale.

```python
from data import get_prices
from algorithm import MovingAverageCrossover
from backtest import Backtester, build_report

prices = get_prices(["AAPL"], "2018-01-01", "2023-01-01")
result = Backtester().run(MovingAverageCrossover("AAPL", short_window=20, long_window=100), prices)

report = build_report(result)
print(report.stats)          # total_return, cagr, sharpe, max_drawdown, ...
report.figures[0].show()     # equity curve (Plotly)
```

## Install

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev,notebook]"
```

## The `Algorithm` contract

```python
class Algorithm(ABC):
    def generate_target_weights(self, prices: pd.DataFrame) -> pd.DataFrame: ...
```

- **In**: `prices` — adjusted close, one column per ticker, no gaps/NaNs.
- **Out**: target fraction of portfolio equity per ticker, same date index as `prices`, values in `[-1, 1]`, `sum(abs(weights)) <= 1` on every date, `0.0` (not `NaN`) during warm-up.

That's the whole contract — the strategy never touches execution, fills, or costs. Copy a built-in algorithm's pattern to write your own (see [notebook/04_custom_algorithm.ipynb](notebook/04_custom_algorithm.ipynb)); no registration step required.

## Built-in algorithms

| Algorithm | Module | Idea |
|---|---|---|
| `MovingAverageCrossover` | `algorithm.ma_crossover` | Long a single stock when short MA > long MA. |
| `PairsTrading` | `algorithm.pairs_trading` | Trade the z-score of a rolling-hedge-ratio spread between two stocks. |
| `PortfolioRebalance` | `algorithm.portfolio_rebalance` | Hold N stocks at fixed target weights, reset on a calendar schedule. |

## Repository layout

```
pyTrader/
  algorithm/     # Algorithm base class + built-in strategies
  data/          # price loader (yfinance) + local parquet cache
  backtest/      # engine, metrics, plotly-based report
  dashboard/     # interactive Dash app
  mcp_server/    # thin MCP tool wrappers
  notebook/      # example notebooks
  tests/
  doc/design.md
```

## Backtest engine

`Backtester.run(algorithm, prices, initial_capital=100_000, commission_bps=0, slippage_bps=0)` is a vectorized, deterministic daily-bar simulator (not an order/fill simulator): it shifts weights by one day to prevent look-ahead bias, charges costs on turnover, and compounds the result into an equity curve. It raises a clear error if a strategy's weights violate the leverage cap or contain NaNs, rather than silently clipping or filling. Everything downstream — metrics, charts, the dashboard, the MCP server — reads from the single `BacktestResult` it returns.

## Dashboard

```bash
python -m dashboard.app
```

Opens a local Dash server (default `http://127.0.0.1:8050`) with form controls for algorithm/tickers/date range/parameters. It calls `Backtester.run()` and `build_report()` directly — no separate backtest logic.

## MCP server

```bash
python -m mcp_server.server
```

Thin tool wrappers with no business logic of their own: `list_algorithms`, `run_backtest`, `get_backtest_report`, `get_current_signal` (today's target weights only, no simulation — the seam a future live broker integration would call).

## Notebooks

`notebook/` follows one pattern throughout — load data, instantiate an algorithm, run the backtest, print stats, show charts — so they double as living API docs:

1. [01_ma_crossover.ipynb](notebook/01_ma_crossover.ipynb)
2. [02_pairs_trading.ipynb](notebook/02_pairs_trading.ipynb)
3. [03_portfolio_rebalance.ipynb](notebook/03_portfolio_rebalance.ipynb)
4. [04_custom_algorithm.ipynb](notebook/04_custom_algorithm.ipynb) — subclassing `Algorithm` from scratch

## Tests

```bash
pytest
```

Each algorithm's signal logic is tested against small synthetic price series with hand-checkable expected output; the engine is tested against a toy example whose equity curve is computed by hand.

## Not doing (yet)

Live trading, intraday/options/futures data, non-US markets, a database, auth. Local files + Python only. A future live broker adapter (e.g. Robinhood) would be driven by `get_current_signal`'s output without changing `algorithm/` or `backtest/`.
