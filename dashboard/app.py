"""Local Dash UI: pick an algorithm/tickers/dates/params, run, see the Report figures.

No backtest logic lives here — it only calls Backtester.run() and build_report(),
the same functions notebooks and the MCP server use.
"""
import json
from datetime import date

from dash import Dash, Input, Output, State, dcc, html

from algorithm import MovingAverageCrossover, PairsTrading, PortfolioRebalance
from backtest import Backtester, build_report
from data import get_prices

ALGORITHMS = {
    "Moving Average Crossover": (MovingAverageCrossover, {"ticker": "AAPL", "short_window": 20, "long_window": 100}),
    "Pairs Trading": (PairsTrading, {"ticker_a": "KO", "ticker_b": "PEP", "window": 60, "entry_z": 2.0, "exit_z": 0.5}),
    "Portfolio Rebalance": (PortfolioRebalance, {"target_weights": {"AAPL": 0.5, "MSFT": 0.5}, "frequency": "ME"}),
}

# Plain-language tooltip for each stat card, plus whether the sign of the
# value should be read as good (green) or bad (red). None = neutral metric.
METRIC_META = {
    "total_return": ("Total Return", "Overall gain or loss over the whole backtest period.", True),
    "cagr": ("CAGR", "Compound Annual Growth Rate -- the annualized return.", True),
    "annual_volatility": ("Volatility", "How much returns swing around, annualized. Higher = bumpier ride.", None),
    "sharpe": ("Sharpe", "Return per unit of risk. Above ~1 is decent, above ~2 is very good.", True),
    "sortino": ("Sortino", "Like Sharpe, but only penalizes downside swings, not all volatility.", True),
    "max_drawdown": ("Max Drawdown", "Worst peak-to-trough loss you'd have sat through. Closer to 0% is better.", True),
    "calmar": ("Calmar", "CAGR earned per unit of max drawdown -- return vs. worst-case pain.", True),
    "win_rate": ("Win Rate", "Share of trading days that were profitable.", None),
    "trade_count": ("Trades", "Number of times the strategy changed its positions.", None),
    "benchmark_total_return": ("Benchmark Return", "What buying and holding the benchmark instead would have returned.", None),
    "benchmark_cagr": ("Benchmark CAGR", "The benchmark's annualized return, for comparison.", None),
}
_PERCENT_KEYS = {"total_return", "cagr", "annual_volatility", "max_drawdown", "win_rate",
                  "benchmark_total_return", "benchmark_cagr"}


def _stat_card(key: str, value) -> html.Div:
    label, tooltip, signed = METRIC_META.get(key, (key, "", None))
    text = str(value) if key == "trade_count" else f"{value * 100:,.1f}%" if key in _PERCENT_KEYS else f"{value:,.2f}"
    direction = "" if signed is None or value == 0 else "positive" if value > 0 else "negative"
    return html.Div(className="stat-card", **{"data-tooltip": tooltip}, children=[
        html.Span(label, className="stat-label"),
        html.Span(text, className=f"stat-value {direction}".strip()),
    ])


def _field(label: str, control) -> html.Div:
    return html.Div(className="field", children=[html.Label(label), control])


app = Dash(__name__)
app.title = "pyTrader"
app.layout = html.Div(
    style={"maxWidth": "900px", "margin": "30px auto", "padding": "0 20px"},
    children=[
        html.Div(className="header", children=[
            html.H2("pyTrader"),
            html.P("Pick a strategy, run a backtest, see how it would have performed."),
        ]),
        html.Div(className="card", children=[
            html.Div(className="controls-grid", children=[
                _field("Algorithm", dcc.Dropdown(id="algorithm", options=list(ALGORITHMS), value="Moving Average Crossover")),
                _field("Tickers (comma-separated)", dcc.Input(id="tickers", value="AAPL", type="text", style={"width": "100%"})),
                _field("Start date", dcc.DatePickerSingle(id="start", date=date(2020, 1, 1))),
                _field("End date", dcc.DatePickerSingle(id="end", date=date.today())),
                _field("Initial capital", dcc.Input(id="capital", value=100_000, type="number")),
                _field("Commission (bps)", dcc.Input(id="commission", value=0, type="number")),
            ]),
            _field("Algorithm parameters (JSON)", dcc.Textarea(id="params", style={"width": "100%", "height": "90px"})),
            html.Button("Run Backtest", id="run", n_clicks=0),
        ]),
        dcc.Loading(children=[
            html.Div(id="stats", className="stats-grid"),
            html.Div(id="figures"),
        ]),
    ],
)


@app.callback(Output("params", "value"), Input("algorithm", "value"))
def _default_params(algorithm_name):
    _, defaults = ALGORITHMS[algorithm_name]
    return json.dumps(defaults, indent=2)


@app.callback(
    Output("stats", "children"),
    Output("figures", "children"),
    Input("run", "n_clicks"),
    State("algorithm", "value"),
    State("tickers", "value"),
    State("start", "date"),
    State("end", "date"),
    State("capital", "value"),
    State("commission", "value"),
    State("params", "value"),
    prevent_initial_call=True,
)
def _run_backtest(_, algorithm_name, tickers, start, end, capital, commission, params_json):
    try:
        params = json.loads(params_json)
        # Portfolio Rebalance already names its tickers as target_weights keys;
        # using those (instead of the separate free-text field) avoids the two
        # falling out of sync.
        tickers = list(params["target_weights"]) if "target_weights" in params else [
            t.strip() for t in tickers.split(",") if t.strip()
        ]
        prices = get_prices(tickers, start, end)

        algo_cls, _ = ALGORITHMS[algorithm_name]
        algorithm = algo_cls(**params)
        result = Backtester().run(algorithm, prices, initial_capital=capital, commission_bps=commission)
        report = build_report(result)
    except (RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return html.Div(f"Error: {exc}", style={"color": "crimson"}), []

    stats = [_stat_card(k, v) for k, v in report.stats.items()]
    figures = [html.Div(className="card", children=dcc.Graph(figure=fig)) for fig in report.figures]
    return stats, figures


if __name__ == "__main__":
    app.run(debug=True)
