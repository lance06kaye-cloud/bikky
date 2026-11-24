"""Plotly charts, shared by the notebook and the future Streamlit app."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import optimize, stats

from promo_incrementality.estimators import EventStudy, weekly_panel
from promo_incrementality.schema import PromoConfig

from promo_incrementality.theme import (
    ACCENT as TREATED_COLOR,
    ACCENT_WASH as PROMO_WASH,
    CONTROL as CONTROL_COLOR,
    GRID,
    MUTED_TEXT,
    NEUTRAL_WASH as POST_WASH,
    SURFACE,
    TEXT,
    TYPE,
)


def _style(fig: go.Figure, title: str, x_title: str, y_title: str | None, x_range: list) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": TYPE["md"], "color": TEXT}, "x": 0, "xanchor": "left"},
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font={"color": MUTED_TEXT, "size": TYPE["xs"]},
        hovermode="x unified",
        margin={"l": 60, "r": 110, "t": 100, "b": 50},
        legend={"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"},
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, zeroline=False, dtick=4, range=x_range)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False)
    return fig


def _shade_promo_windows(fig: go.Figure, config: PromoConfig) -> None:
    """Shade the promo and post windows. Weeks are centered on their tick marks."""
    promo_end = config.promo_weeks - 0.5
    post_end = promo_end + config.post_window_weeks
    fig.add_vrect(x0=-0.5, x1=promo_end, fillcolor=PROMO_WASH, line_width=0, layer="below",
                  annotation_text="Promo", annotation_position="top left")
    fig.add_vrect(x0=promo_end, x1=post_end, fillcolor=POST_WASH, line_width=0, layer="below",
                  annotation_text="Post window", annotation_position="top left")


def weekly_visits_chart(
    transactions: pd.DataFrame, config: PromoConfig, panel: pd.DataFrame | None = None
) -> go.Figure:
    """Mean weekly visits per location, treated vs. control."""
    panel = weekly_panel(transactions, config) if panel is None else panel
    return weekly_means_chart(weekly_means(panel), config)


def weekly_means(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean visits per location by week. Columns: 1 for treated, 0 for control."""
    return panel.groupby(["week", "treated"])["visits"].mean().unstack()


def weekly_means_chart(means: pd.DataFrame, config: PromoConfig) -> go.Figure:
    """Line chart of weekly means. Columns are 1 for treated and 0 for control."""
    fig = go.Figure()
    for column, name, color in [(1, "Promo markets", TREATED_COLOR),
                                (0, "Control markets", CONTROL_COLOR)]:
        fig.add_trace(
            go.Scatter(
                x=means.index,
                y=means[column],
                name=name,
                mode="lines",
                line={"color": color, "width": 2},
                hovertemplate="%{y:,.0f}<extra>" + name + "</extra>",
            )
        )
        # Nudge the labels apart, because the two lines can end very close together.
        is_upper = means[column].iloc[-1] >= means[1 - column].iloc[-1]
        fig.add_annotation(x=means.index[-1], y=means[column].iloc[-1], text=name, showarrow=False,
                           xanchor="left", xshift=6, yshift=8 if is_upper else -8,
                           font={"color": TEXT, "size": TYPE["xs"]})
    _shade_promo_windows(fig, config)
    x_range = [means.index[0] - 0.5, means.index[-1] + 0.5]
    return _style(fig, "Weekly visits per location", "Weeks from promo start", "Visits", x_range)


def event_study_chart(study: EventStudy, config: PromoConfig, title: str | None = None) -> go.Figure:
    """Week-by-week treated-vs-control gap with 95% bars."""
    data = study.coefficients
    fig = go.Figure(
        go.Scatter(
            x=data["week"],
            y=data["estimate"],
            mode="lines+markers",
            line={"color": TREATED_COLOR, "width": 2},
            marker={"color": TREATED_COLOR, "size": 8},
            error_y={
                "type": "data",
                "array": data["ci_high"] - data["estimate"],
                "arrayminus": data["estimate"] - data["ci_low"],
                "color": TREATED_COLOR,
                "thickness": 1,
                "width": 0,
            },
            hovertemplate="%{y:+.1f} visits<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line_color=MUTED_TEXT, line_width=1)
    _shade_promo_windows(fig, config)
    title = title or f"Promo effect by week (pre-trend drift test p = {study.pretrend_p_value:.2f})"
    x_range = [data["week"].min() - 0.5, data["week"].max() + 0.5]
    return _style(fig, title, "Weeks from promo start", "Visits per location vs. week -1", x_range)


def density_from_samples(values) -> tuple:
    """Smooth a set of outcomes into (x, density) with a kernel density estimate.

    Raises ValueError if fewer than two distinct finite values are given.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(np.unique(values)) < 2:
        raise ValueError("need at least two distinct values to draw a distribution")
    pad = 0.15 * (values.max() - values.min())
    x = np.linspace(values.min() - pad, values.max() + pad, 200)
    return x, stats.gaussian_kde(values)(x)


def interval_sds(estimate: float, low: float, high: float, coverage: float = 0.95) -> tuple:
    """Left and right SDs of a two-piece normal that peaks at the estimate.

    Bootstrap ranges are often lopsided around the estimate, so each side gets its
    own width. Both halves share one peak height, so the curve is smooth at the
    estimate, and the widths are solved so `low` and `high` cut off equal tails,
    (1 - coverage) / 2 each. A range that doesn't contain the estimate falls back to
    one symmetric width. A very lopsided range, with one side many times the other,
    has no exact fit, so it gets the closest one: its tails come out a little off.
    """
    tail = (1 - coverage) / 2
    z = stats.norm.ppf(1 - tail)
    if not low < estimate < high:
        sd = (high - low) / (2 * z)
        return sd, sd
    below, above = estimate - low, high - estimate

    def tail_errors(log_sds):
        left, right = np.exp(log_sds)
        return [
            2 * left / (left + right) * stats.norm.cdf(-below / left) - tail,
            2 * right / (left + right) * stats.norm.cdf(-above / right) - tail,
        ]

    # Errors are relative to the tail, so a closest fit is judged the same on both sides
    solution = optimize.least_squares(
        lambda log_sds: np.array(tail_errors(log_sds)) / tail,
        np.log([below / z, above / z]),
        xtol=1e-12, ftol=1e-12,
    )
    left, right = np.exp(solution.x)
    return left, right


def _two_piece_cdf(value: float, estimate: float, left: float, right: float) -> float:
    """Share of the two-piece normal below `value`."""
    if value < estimate:
        return 2 * left / (left + right) * stats.norm.cdf((value - estimate) / left)
    return 1 - 2 * right / (left + right) * stats.norm.sf((value - estimate) / right)


def density_from_interval(
    estimate: float, low: float, high: float, coverage: float = 0.95
) -> tuple:
    """Approximate (x, density) as a smooth curve whose middle `coverage` spans low to high."""
    left, right = interval_sds(estimate, low, high, coverage)
    x = np.linspace(estimate - 4 * left, estimate + 4 * right, 200)
    peak = 2 / (np.sqrt(2 * np.pi) * (left + right))
    width = np.where(x < estimate, left, right)
    return x, peak * np.exp(-0.5 * ((x - estimate) / width) ** 2)


def chance_above(
    value: float, estimate: float, low: float, high: float, coverage: float = 0.95
) -> float:
    """Chance of landing above `value` under the density_from_interval curve."""
    left, right = interval_sds(estimate, low, high, coverage)
    return 1 - _two_piece_cdf(value, estimate, left, right)


def chance_step(low: float, high: float, steps: int = 40) -> float:
    """A round step (1, 2 or 5 times a power of ten) that cuts low..high into about `steps` pieces."""
    rough = (high - low) / steps
    power = 10 ** np.floor(np.log10(rough))
    return min((m * power for m in (1, 2, 5, 10) if m * power >= rough), default=10 * power)


def density_chart(
    x, density, title: str, x_title: str, markers: list, value_format: str = ",.1f"
) -> go.Figure:
    """Probability density of an outcome. `markers` are (value, label) reference lines.

    Markers without a finite value are left out. Neighbouring labels sit on opposite
    sides of their lines and at different heights, so they stay readable even when
    the lines nearly touch. `value_format` is a d3 format for the axis and hover,
    e.g. "$,.0f" for whole dollars.

    The y axis is a percentage: the chance of landing within one round step of x
    (density times step). The axis has no title, just % ticks.
    """
    markers = [(value, label) for value, label in markers if np.isfinite(value)]
    step = chance_step(x[0], x[-1])
    percent = 100 * np.asarray(density) * step
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=percent,
            mode="lines",
            line={"color": TREATED_COLOR, "width": 2},
            fill="tozeroy",
            fillcolor=PROMO_WASH,
            hovertemplate="%{x:" + value_format + "}: %{y:.1f}% chance<extra></extra>",
            showlegend=False,
        )
    )
    ordered = sorted(markers)
    for position, (value, label) in enumerate(ordered):
        fig.add_vline(
            x=value,
            line_color=MUTED_TEXT,
            line_width=1,
            annotation_text=label,
            annotation_position="top left" if position % 2 == 0 else "top right",
            annotation_yshift=0 if position % 2 == 0 else -18,
        )
    x_low = min([x[0]] + [value for value, _ in markers])
    x_high = max([x[-1]] + [value for value, _ in markers])
    pad = 0.05 * (x_high - x_low)
    _style(fig, title, x_title, None, [x_low - pad, x_high + pad])
    fig.update_xaxes(dtick=None, tickformat=value_format)
    # Headroom, so the reference-line labels sit above the curve's peak
    fig.update_yaxes(ticksuffix="%", rangemode="tozero", range=[0, 1.2 * np.max(percent)])
    return fig
