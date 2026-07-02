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

app = Dash(__name__)
app.layout = html.Div(
    style={"maxWidth": "700px", "margin": "auto", "fontFamily": "sans-serif"},
    children=[
        html.H2("pyTrader"),
        html.Label("Algorithm"),
        dcc.Dropdown(id="algorithm", options=list(ALGORITHMS), value="Moving Average Crossover"),
        html.Label("Tickers (comma-separated)"),
        dcc.Input(id="tickers", value="AAPL", type="text", style={"width": "100%"}),
        html.Label("Start date"),
        dcc.DatePickerSingle(id="start", date=date(2020, 1, 1)),
        html.Label("End date"),
        dcc.DatePickerSingle(id="end", date=date.today()),
        html.Label("Initial capital"),
        dcc.Input(id="capital", value=100_000, type="number"),
        html.Label("Commission (bps)"),
        dcc.Input(id="commission", value=0, type="number"),
        html.Label("Algorithm parameters (JSON)"),
        dcc.Textarea(id="params", style={"width": "100%", "height": "100px"}),
        html.Button("Run", id="run", n_clicks=0),
        html.Div(id="stats"),
        html.Div(id="figures"),
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

    stats = html.Ul([html.Li(f"{k}: {v:,.4f}" if isinstance(v, float) else f"{k}: {v}") for k, v in report.stats.items()])
    figures = [dcc.Graph(figure=fig) for fig in report.figures]
    return stats, figures


if __name__ == "__main__":
    app.run(debug=True)
