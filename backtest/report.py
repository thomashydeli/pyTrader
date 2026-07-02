from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

from .engine import BacktestResult
from .metrics import compute_metrics

# Shared look for every chart -- dark, transparent background so it blends into
# the dashboard, plus a subtitle slot so each figure can explain itself in one line.
_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif", color="#e6e6e6"),
    hovermode="x unified",
    margin=dict(t=56, l=50, r=30, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
_GREEN, _RED, _GRAY = "#00c805", "#ff453a", "#8f98a3"


def _titled(fig: go.Figure, title: str, subtitle: str, yaxis_title: str) -> go.Figure:
    """Apply the shared theme plus a short plain-language subtitle explaining the chart."""
    fig.update_layout(
        **_LAYOUT,
        title=dict(text=f"{title}<br><span style='font-size:0.7em;color:{_GRAY}'>{subtitle}</span>"),
        xaxis_title="Date",
        yaxis_title=yaxis_title,
    )
    return fig


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
    fig = go.Figure(go.Scatter(x=result.equity.index, y=result.equity, name="Strategy", line=dict(color=_GREEN)))
    if benchmark_prices is not None:
        bench_equity = result.initial_capital * (
            1 + benchmark_prices.pct_change().reindex(result.equity.index).fillna(0.0)
        ).cumprod()
        fig.add_scatter(x=bench_equity.index, y=bench_equity, name="Benchmark", line=dict(color=_GRAY, dash="dot"))
    return _titled(fig, "Equity Curve", "How your account balance would have grown over time", "Equity ($)")


def drawdown_chart(result: BacktestResult) -> go.Figure:
    drawdown = result.equity / result.equity.cummax() - 1
    fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", name="Drawdown",
                                line=dict(color=_RED)))
    return _titled(fig, "Drawdown", "Loss from the highest point so far -- closer to 0% is better", "Drawdown")


def weights_area_chart(result: BacktestResult) -> go.Figure:
    fig = go.Figure()
    for ticker in result.weights.columns:
        fig.add_scatter(x=result.weights.index, y=result.weights[ticker], name=ticker, stackgroup="weights")
    return _titled(fig, "Target Weights Over Time", "How the portfolio is split across holdings each day", "Weight")


def price_signal_chart(result: BacktestResult, ticker: str) -> go.Figure:
    # Based on the direction of each weight change (not "== 0"), so a direct
    # long<->short flip is marked too, not just transitions through flat.
    price, weight = result.prices[ticker], result.weights[ticker]
    change = weight - weight.shift(1).fillna(0.0)
    entries = weight[change > 0]
    exits = weight[change < 0]

    fig = go.Figure(go.Scatter(x=price.index, y=price, name=ticker, line=dict(color=_GRAY)))
    fig.add_scatter(x=entries.index, y=price.loc[entries.index], mode="markers", name="Entry",
                     marker=dict(symbol="triangle-up", color=_GREEN, size=10))
    fig.add_scatter(x=exits.index, y=price.loc[exits.index], mode="markers", name="Exit",
                     marker=dict(symbol="triangle-down", color=_RED, size=10))
    return _titled(fig, f"{ticker} Price & Signals", "▲ where the strategy buys, ▼ where it sells", "Price")


def pairs_spread_chart(spread: pd.Series, zscore: pd.Series, entry_z: float, exit_z: float) -> go.Figure:
    fig = go.Figure(go.Scatter(x=zscore.index, y=zscore, name="Z-score"))
    for level, label in ((entry_z, "Entry"), (-entry_z, None), (exit_z, "Exit"), (-exit_z, None)):
        fig.add_hline(y=level, line_dash="dash", annotation_text=label)
    return _titled(fig, "Pairs Spread Z-score",
                   "Trades when the spread strays far from its average, exits as it reverts", "Z-score")
