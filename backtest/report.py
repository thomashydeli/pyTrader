from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

from .engine import BacktestResult
from .metrics import compute_metrics


@dataclass
class Report:
    stats: dict
    figures: list[go.Figure]


def build_report(result: BacktestResult, benchmark_prices: pd.Series | None = None) -> Report:
    """One entry point shared by notebooks, the dashboard, and the MCP server."""
    figures = [
        equity_curve_chart(result, benchmark_prices),
        drawdown_chart(result),
        weights_area_chart(result),
    ]
    if result.weights.shape[1] == 1:
        figures.append(price_signal_chart(result, result.weights.columns[0]))
    return Report(stats=compute_metrics(result, benchmark_prices), figures=figures)


def equity_curve_chart(result: BacktestResult, benchmark_prices: pd.Series | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=result.equity.index, y=result.equity, name="Strategy"))
    if benchmark_prices is not None:
        bench_equity = result.initial_capital * (
            1 + benchmark_prices.pct_change().reindex(result.equity.index).fillna(0.0)
        ).cumprod()
        fig.add_scatter(x=bench_equity.index, y=bench_equity, name="Benchmark")
    fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Equity ($)")
    return fig


def drawdown_chart(result: BacktestResult) -> go.Figure:
    drawdown = result.equity / result.equity.cummax() - 1
    fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", name="Drawdown"))
    fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown")
    return fig


def weights_area_chart(result: BacktestResult) -> go.Figure:
    fig = go.Figure()
    for ticker in result.weights.columns:
        fig.add_scatter(x=result.weights.index, y=result.weights[ticker], name=ticker, stackgroup="weights")
    fig.update_layout(title="Target Weights Over Time", xaxis_title="Date", yaxis_title="Weight")
    return fig


def price_signal_chart(result: BacktestResult, ticker: str) -> go.Figure:
    # Based on the direction of each weight change (not "== 0"), so a direct
    # long<->short flip is marked too, not just transitions through flat.
    price, weight = result.prices[ticker], result.weights[ticker]
    change = weight - weight.shift(1).fillna(0.0)
    entries = weight[change > 0]
    exits = weight[change < 0]

    fig = go.Figure(go.Scatter(x=price.index, y=price, name=ticker))
    fig.add_scatter(x=entries.index, y=price.loc[entries.index], mode="markers", name="Entry",
                     marker=dict(symbol="triangle-up", color="green", size=10))
    fig.add_scatter(x=exits.index, y=price.loc[exits.index], mode="markers", name="Exit",
                     marker=dict(symbol="triangle-down", color="red", size=10))
    fig.update_layout(title=f"{ticker} Price & Signals", xaxis_title="Date", yaxis_title="Price")
    return fig


def pairs_spread_chart(spread: pd.Series, zscore: pd.Series, entry_z: float, exit_z: float) -> go.Figure:
    fig = go.Figure(go.Scatter(x=zscore.index, y=zscore, name="Z-score"))
    for level, label in ((entry_z, "Entry"), (-entry_z, None), (exit_z, "Exit"), (-exit_z, None)):
        fig.add_hline(y=level, line_dash="dash", annotation_text=label)
    fig.update_layout(title="Pairs Spread Z-score", xaxis_title="Date", yaxis_title="Z-score")
    return fig
