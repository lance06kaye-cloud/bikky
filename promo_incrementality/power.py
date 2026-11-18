"""Before running a promo: what is the smallest effect this test could detect?

The calculator replays fake promos on promo-free history. Each placebo picks a
start week and a set of markets, then measures the "effect" the same way `did`
does. No promo happened, so the spread of those effects is the noise a real test
would face. It captures whatever noise the history actually has, including market
shocks, with no assumptions about its shape.
"""

import numpy as np
import pandas as pd
from scipy import stats

from promo_incrementality.estimators import clusters_by_market
from promo_incrementality.schema import validate_transactions
from promo_incrementality.synth import pick_treated_markets


class PowerError(ValueError):
    """The history can't support a placebo-based power estimate."""


def full_weeks(history: pd.DataFrame) -> int:
    """How many full weeks `weekly_visits` will find in this history."""
    timestamps = history["timestamp"].dt.normalize()
    return ((timestamps.max() - timestamps.min()).days + 1) // 7


def weekly_visits(history: pd.DataFrame) -> pd.DataFrame:
    """Visits by week (rows) and location (columns), with full weeks only."""
    validate_transactions(history)
    days = (history["timestamp"].dt.normalize() - history["timestamp"].min().normalize()).dt.days
    last_full_week = full_weeks(history)
    visits = history.groupby([(days // 7).rename("week"), "location_id"]).size().unstack(fill_value=0)
    return visits.reindex(range(last_full_week), fill_value=0)


def _placebo_effect(visits: np.ndarray, treated: np.ndarray, start: int, n_weeks: int) -> float:
    """The net DiD estimate for a fake promo, in visits.

    `visits` is weeks by locations, `treated` a boolean mask over locations, and the
    weeks before `start` are the baseline. On a balanced panel the two-way fixed
    effects coefficient is exactly this difference in before-and-after means of
    indexed visits. Indexed visits average 1 over the baseline by construction.
    """
    baseline = visits[:start].mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):  # a closed store: not finite
        change = visits[start : start + n_weeks].mean(axis=0) / baseline - 1
    effect = change[treated].mean() - change[~treated].mean()
    return effect * baseline[treated].sum() * n_weeks


def minimum_detectable_effect(
    history: pd.DataFrame,
    location_market: dict,
    promo_weeks: int = 4,
    post_window_weeks: int = 6,
    pre_weeks: int = 24,
    treated_share: float = 0.5,
    power: float = 0.8,
    n_placebos: int = 1000,
    seed: int = 0,
) -> dict:
    """Estimate the smallest net visit lift the planned test could reliably detect.

    `history` must be free of promos. The result is the net effect that a two-sided
    5% test would catch with the given `power`. `pre_weeks` is the baseline the real
    test will have, and every placebo uses exactly that much, because a short
    baseline adds noise of its own. Distinct placebos are limited by the number of
    start weeks and market splits, so short histories give rough answers.
    """
    visits = weekly_visits(history)
    n_weeks = promo_weeks + post_window_weeks
    last_start = len(visits) - n_weeks
    if last_start < pre_weeks:
        raise PowerError(f"history needs at least {pre_weeks + n_weeks} full weeks")

    unmapped = sorted(set(visits.columns) - set(location_market))
    if unmapped:
        raise ValueError(f"no market given for locations: {unmapped}")
    markets = pd.Series(location_market).reindex(visits.columns)
    if not clusters_by_market(location_market, visits.columns):  # the same rule as did()
        markets = pd.Series(visits.columns, index=visits.columns)
    market_ids = sorted(markets.unique())
    market_sizes = [int((markets == market).sum()) for market in market_ids]

    rng = np.random.default_rng(seed)
    values = visits.to_numpy(dtype=float)
    location_market = markets.to_numpy()
    effects = []
    for _ in range(n_placebos):
        chosen = pick_treated_markets(market_sizes, treated_share, rng)
        treated = np.isin(location_market, [market_ids[i] for i in chosen])
        start = int(rng.integers(pre_weeks, last_start + 1))
        effects.append(_placebo_effect(values[start - pre_weeks :], treated, pre_weeks, n_weeks))

    effects = np.array(effects)
    usable = effects[np.isfinite(effects)]  # a store with no baseline visits can't be indexed
    if len(usable) < 20:
        raise PowerError("too few placebo windows where every location had baseline visits")
    noise = float(np.std(usable, ddof=1))
    multiplier = stats.norm.ppf(0.975) + stats.norm.ppf(power)
    mde = multiplier * noise
    weekly_treated_visits = visits.mean().mean() * len(visits.columns) * treated_share
    return {
        "promo_weeks": promo_weeks,
        "post_window_weeks": post_window_weeks,
        "noise_sd_visits": noise,
        "min_detectable_visits": mde,
        "min_detectable_lift": mde / (weekly_treated_visits * n_weeks),
    }


def design_table(history: pd.DataFrame, location_market: dict, designs: list, **kwargs) -> pd.DataFrame:
    """Compare (promo_weeks, post_window_weeks) designs on the same history."""
    return pd.DataFrame(
        minimum_detectable_effect(
            history, location_market, promo_weeks=promo, post_window_weeks=post, **kwargs
        )
        for promo, post in designs
    )
