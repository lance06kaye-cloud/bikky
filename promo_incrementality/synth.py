"""Synthetic restaurant transactions with known, planted promo effects.

Every visit is built from labelled parts, so the counterfactual is exact rather
than estimated:

    baseline     would have happened anyway (some redeem the discount: subsidized)
    incremental  caused by the promo (always redeem)
    halo         extra post-promo visits from guests the promo won or won back

Pull-forward deletes real baseline visits from the weeks after the promo, so the
true net effect is `incremental - pull_forward + halo`. The labels go into
`ground_truth` only. The transactions look like what a POS would export.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from promo_incrementality.schema import TRANSACTION_COLUMNS, PromoConfig

SEGMENTS = ["regular", "occasional", "lapsed", "new"]
PROMO_CODE = "SUMMER"  # the sample config's promo_codes depends on this


@dataclass
class SynthConfig:
    seed: int = 42

    # Network: locations per market. 20 locations in 5 markets.
    market_sizes: tuple = (5, 5, 4, 3, 3)
    location_size_sigma: float = 0.25  # lognormal spread of guests per location

    # Guests
    n_guests: int = 50_000
    segment_shares: dict = field(
        default_factory=lambda: {"regular": 0.15, "occasional": 0.35, "lapsed": 0.30, "new": 0.20}
    )
    # Mean visits per week. "new" guests start at this rate when they arrive.
    segment_visit_rates: dict = field(
        default_factory=lambda: {"regular": 1.0, "occasional": 0.25, "lapsed": 0.03, "new": 0.30}
    )
    rate_dispersion: float = 2.0  # gamma shape: guests vary around their segment rate

    # Calendar: time effects shared by every location
    start_date: str = "2024-01-01"
    n_weeks: int = 52
    seasonality_amplitude: float = 0.15
    seasonality_peak_week: int = 32
    annual_trend: float = 0.08
    # Weekly market-level demand shocks (share of demand). Zero means location
    # clusters are independent; see the clustering note in the README.
    market_shock_sd: float = 0.0
    # Week-to-week correlation of those shocks. Zero means independent weeks.
    market_shock_persistence: float = 0.0

    # Promo design: assigned by market, starting on the upswing of the season
    treated_location_share: float = 0.5
    promo_start_week: int = 24
    promo_weeks: int = 4
    post_window_weeks: int = 6
    discount_amount: float = 3.0
    avg_order_value: float = 18.0
    avg_margin_per_order: float = 6.0

    # Planted true effects
    # Proportional lift in visit rate during the promo.
    lift_by_segment: dict = field(
        default_factory=lambda: {"regular": 0.0, "occasional": 0.10, "lapsed": 1.0, "new": 0.25}
    )
    # Share of baseline promo-period visits, in any segment, that redeem anyway.
    subsidized_share: float = 0.20
    # Share of incremental visits, by segment, that try to borrow a visit from the
    # same guest's following weeks. Lapsed guests have almost nothing to borrow.
    # A guest with no later visit borrows nothing, so the realized share is lower;
    # ground_truth reports it as pull_forward_realized_share.
    pull_forward_attempt_share: dict = field(
        default_factory=lambda: {"regular": 0.5, "occasional": 0.5, "lapsed": 0.0, "new": 0.5}
    )
    pull_forward_weeks: int = 4
    # Extra weekly visits, from promo end onward, for lapsed or new guests who
    # made an incremental visit.
    halo_weekly_rate: float = 0.05


@dataclass
class SynthResult:
    transactions: pd.DataFrame
    ground_truth: dict
    promo_config: PromoConfig
    location_market: dict  # location_id -> market_id


def _weekly_demand(config: SynthConfig) -> np.ndarray:
    """Shared demand multiplier by week: seasonality times a mild upward trend."""
    weeks = np.arange(config.n_weeks)
    season = 1 + config.seasonality_amplitude * np.cos(
        2 * np.pi * (weeks - config.seasonality_peak_week) / 52
    )
    return season * (1 + config.annual_trend * weeks / 52)


def pick_treated_markets(market_sizes, share: float, rng: np.random.Generator) -> list:
    """Add markets in random order until the location share is as close as it gets.

    Always treats at least one market and leaves at least one as control.
    """
    target = share * sum(market_sizes)
    order = rng.permutation(len(market_sizes))
    treated, n_treated = [int(order[0])], market_sizes[order[0]]
    for market in order[1:-1]:
        size = market_sizes[market]
        if abs(n_treated + size - target) >= abs(n_treated - target):
            break
        treated.append(int(market))
        n_treated += size
    return treated


def _market_shocks(config: SynthConfig, rng: np.random.Generator) -> np.ndarray:
    """Demand multipliers by market and week. Their sd ignores persistence."""
    noise = rng.standard_normal((len(config.market_sizes), config.n_weeks))
    rho = config.market_shock_persistence
    shocks = noise.copy()
    for week in range(1, config.n_weeks):
        shocks[:, week] = rho * shocks[:, week - 1] + np.sqrt(1 - rho**2) * noise[:, week]
    return 1 + config.market_shock_sd * shocks


def _take_pull_forward(baseline, wanted, weeks) -> np.ndarray:
    """Delete up to `wanted` baseline visits per guest, earliest week first.

    Edits `baseline` in place and returns the visits deleted, by week (columns
    follow `weeks`).
    """
    remaining = wanted.copy()
    taken_by_week = []
    for week in weeks:
        taken = np.minimum(remaining, baseline[:, week])
        baseline[:, week] -= taken
        remaining -= taken
        taken_by_week.append(taken.sum())
    return np.array(taken_by_week)


def _to_rows(counts, week_start, guest_location, config, rng, redeemed) -> pd.DataFrame:
    """Expand a guest-by-week count matrix into one row per visit."""
    guests, weeks = np.nonzero(counts)
    # np.repeat wants native-size counts; in the browser (32-bit) int64 is refused
    repeats = counts[guests, weeks].astype(np.intp)
    guests, weeks = np.repeat(guests, repeats), np.repeat(weeks, repeats)
    n = len(guests)

    hours_into_week = rng.integers(0, 7, n) * 24 + rng.integers(7, 22, n)
    timestamp = week_start[weeks] + pd.to_timedelta(hours_into_week, unit="h")
    # Lognormal order size whose mean is avg_order_value.
    sigma = 0.4
    order_total = rng.lognormal(np.log(config.avg_order_value) - sigma**2 / 2, sigma, n)

    return pd.DataFrame(
        {
            "guest_id": guests,
            "location_id": guest_location[guests],
            "timestamp": timestamp,
            "order_total": order_total.round(2),
            "discount_amount": config.discount_amount if redeemed else 0.0,
            "promo_code": PROMO_CODE if redeemed else None,
            "channel": rng.choice(["dine_in", "pickup", "delivery"], n, p=[0.5, 0.3, 0.2]),
        }
    )


def generate(config: SynthConfig | None = None) -> SynthResult:
    """Simulate one year of transactions and return them with the planted truth."""
    config = config or SynthConfig()
    rng = np.random.default_rng(config.seed)

    # Network
    market_of_location = np.repeat(np.arange(len(config.market_sizes)), config.market_sizes)
    n_locations = len(market_of_location)
    location_ids = np.array([f"L{i + 1:02d}" for i in range(n_locations)])
    location_market = {
        str(loc): f"M{market + 1}" for loc, market in zip(location_ids, market_of_location)
    }
    treated_markets = pick_treated_markets(
        config.market_sizes, config.treated_location_share, rng
    )
    treated_location = np.isin(market_of_location, treated_markets)

    # Guests
    n = config.n_guests
    segment = rng.choice(SEGMENTS, n, p=[config.segment_shares[s] for s in SEGMENTS])
    location_weights = rng.lognormal(0, config.location_size_sigma, n_locations)
    home = rng.choice(n_locations, n, p=location_weights / location_weights.sum())
    guest_rate = np.array([config.segment_visit_rates[s] for s in segment]) * rng.gamma(
        config.rate_dispersion, 1 / config.rate_dispersion, n
    )
    # New guests arrive evenly through the year; everyone else is active from week 0.
    arrival_week = np.where(segment == "new", rng.integers(0, config.n_weeks, n), 0)

    # Baseline: the world without the promo
    weeks = np.arange(config.n_weeks)
    market_shock = _market_shocks(config, rng)
    baseline_rate = (
        guest_rate[:, None]
        * _weekly_demand(config)[None, :]
        * market_shock[market_of_location[home]]
        * (weeks[None, :] >= arrival_week[:, None])
    )
    baseline = rng.poisson(baseline_rate)

    # Promo weeks
    promo_slice = slice(config.promo_start_week, config.promo_start_week + config.promo_weeks)
    post_start = promo_slice.stop
    post_end = post_start + config.post_window_weeks
    in_treated = treated_location[home]

    # Incremental visits: extra visits at treated locations during the promo
    lift = np.array([config.lift_by_segment[s] for s in segment])
    incremental = np.zeros_like(baseline)
    incremental[:, promo_slice] = rng.poisson(
        baseline_rate[:, promo_slice] * (lift * in_treated)[:, None]
    )
    incremental_per_guest = incremental.sum(axis=1)

    # Pull-forward: incremental visits replace baseline visits in the following weeks
    attempt_share = np.array([config.pull_forward_attempt_share[s] for s in segment])
    wanted = rng.binomial(incremental_per_guest, attempt_share)
    pull_forward_weeks = range(post_start, min(post_start + config.pull_forward_weeks, config.n_weeks))
    pull_forward = _take_pull_forward(baseline, wanted, pull_forward_weeks)

    # Halo: a small lasting lift for guests the promo acquired or reactivated
    won_guest = (incremental_per_guest > 0) & np.isin(segment, ["lapsed", "new"])
    halo = np.zeros_like(baseline)
    halo[:, post_start:] = rng.poisson(
        config.halo_weekly_rate * won_guest[:, None], (n, config.n_weeks - post_start)
    )

    # Subsidized: baseline promo visits at treated locations that redeem anyway
    subsidized = np.zeros_like(baseline)
    subsidized[:, promo_slice] = rng.binomial(
        baseline[:, promo_slice] * in_treated[:, None], config.subsidized_share
    )
    full_price = baseline - subsidized

    # Transactions
    week_start = pd.Timestamp(config.start_date) + pd.to_timedelta(weeks * 7, unit="D")
    guest_location = location_ids[home]
    parts = [
        _to_rows(full_price + halo, week_start, guest_location, config, rng, redeemed=False),
        _to_rows(subsidized + incremental, week_start, guest_location, config, rng, redeemed=True),
    ]
    transactions = (
        pd.concat(parts, ignore_index=True)[TRANSACTION_COLUMNS]
        .sort_values("timestamp", ignore_index=True)
    )

    promo_config = PromoConfig(
        promo_start=week_start[config.promo_start_week],
        promo_end=week_start[post_start] - pd.Timedelta(days=1),
        treated_location_ids=location_ids[treated_location].tolist(),
        post_window_weeks=config.post_window_weeks,
        avg_margin_per_order=config.avg_margin_per_order,
        location_market=location_market,
    )

    incremental_visits = int(incremental.sum())
    subsidized_redemptions = int(subsidized.sum())
    # Only pull-forward inside the post window is part of the measured net effect.
    in_window = np.array(pull_forward_weeks) < post_end
    pull_forward_visits = int(pull_forward[in_window].sum())
    halo_visits = int(halo[:, post_start:post_end].sum())
    redemptions = incremental_visits + subsidized_redemptions
    ground_truth = {
        "incremental_visits": incremental_visits,
        "subsidized_redemptions": subsidized_redemptions,
        "pull_forward_visits": pull_forward_visits,
        "pull_forward_realized_share": int(pull_forward.sum()) / max(incremental_visits, 1),
        "pull_forward_after_window": int(pull_forward[~in_window].sum()),
        "halo_visits": halo_visits,
        "net_incremental_visits": incremental_visits - pull_forward_visits + halo_visits,
        "post_window_effect": halo_visits - pull_forward_visits,
        "redemptions": redemptions,
        "total_discount_cost": redemptions * config.discount_amount,
        "halo_visits_after_window": int(halo[:, post_end:].sum()),
        "incremental_by_segment": {
            s: int(incremental[segment == s].sum()) for s in SEGMENTS
        },
    }
    return SynthResult(transactions, ground_truth, promo_config, location_market)
