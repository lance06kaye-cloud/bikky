"""Learn simulator settings from an uploaded order history.

`calibrate` turns a business's own orders into `SynthConfig` overrides, so the
simulated view plays out a promo on a chain shaped like theirs: the same stores
and markets, roughly the same guests and visit habits, and the same seasons. If
the file holds a past promo, it also measures that promo's lift and suggests
slider values from it.

Everything here is a rough, documented heuristic. It only has to make the
simulation resemble the business, not estimate anything precisely.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from promo_incrementality.estimators import did, promo_redemptions, weekly_panel
from promo_incrementality.schema import PromoConfig
from promo_incrementality.synth import SynthConfig, generate

# The simulation slows down with more guests, so it runs at most this many
MAX_GUESTS = 60_000
# Visits a week per guest at the default segment mix, for files without guest_id
DEFAULT_VISITS_PER_GUEST = 0.3
# Guests visiting at least this often a week before the promo count as regulars,
# and below LAPSED_RATE as lapsed. In between they are occasional.
REGULAR_RATE = 0.6
LAPSED_RATE = 0.1
# Guests first seen in the later part of the history count as new
NEW_AFTER_SHARE = 0.5
MIN_SEASON_WEEKS = 26
DEFAULT_DISCOUNT = 1.75


@dataclass
class Calibration:
    overrides: dict  # SynthConfig field -> value learned from the data
    margin: float  # average margin per order, from the PromoConfig
    discount: float  # average discount per redemption in the data
    # Suggested slider values in percent from a past promo in the file, or None
    lifts: dict | None
    notes: list = field(default_factory=list)  # plain-English lines for a marketer


def _has(transactions: pd.DataFrame, column: str) -> bool:
    return column in transactions.columns and transactions[column].notna().any()


def _market_sizes(transactions: pd.DataFrame, config: PromoConfig) -> tuple:
    """Stores per market, largest first. Without markets, each store is its own.

    The simulator needs at least two markets, one promo and one control.
    """
    locations = sorted(transactions["location_id"].dropna().unique())
    mapping = config.location_market or {location: location for location in locations}
    sizes = pd.Series([mapping.get(location, location) for location in locations]).value_counts()
    sizes = tuple(sorted((int(size) for size in sizes), reverse=True))
    if len(sizes) < 2:
        # One market: split its stores in two so there is a control group
        total = sizes[0] if sizes else 2
        sizes = (max(total - total // 2, 1), max(total // 2, 1))
    return sizes


def _history_weeks(transactions: pd.DataFrame) -> float:
    span = transactions["timestamp"].max() - transactions["timestamp"].min()
    return max(span.days + 1, 1) / 7


def _guests(transactions: pd.DataFrame, config: PromoConfig) -> tuple:
    """Guest count, segment shares and visit rates, plus notes.

    Each guest's weekly visit rate is measured before the promo, when there are at
    least 4 weeks of it, and over the whole history otherwise. Guests first seen in
    the later half of the history are new, and their rate counts from their first
    visit. The rest are regular (REGULAR_RATE or more), lapsed (under LAPSED_RATE)
    or occasional. The guest count is scaled to a year by the square root of the
    history's length, because a longer history keeps finding new guests but
    ever more slowly.
    """
    defaults = SynthConfig()
    weeks = _history_weeks(transactions)
    notes = []
    if not _has(transactions, "guest_id"):
        visits_per_week = len(transactions) / weeks
        n_guests = visits_per_week / DEFAULT_VISITS_PER_GUEST
        notes.append("Your file has no guest IDs, so guest counts and visit habits are "
                     "estimated from typical restaurants.")
        return n_guests, None, None, notes

    data = transactions.dropna(subset=["guest_id"])
    start = data["timestamp"].min()
    week = (data["timestamp"] - start).dt.days / 7
    first_seen = week.groupby(data["guest_id"]).min()
    pre_weeks = (config.promo_start - start).days / 7
    if pre_weeks >= 4:
        before = data["timestamp"] < config.promo_start
        rate = data[before].groupby("guest_id").size().reindex(first_seen.index, fill_value=0)
        rate = rate / pre_weeks
    else:
        rate = data.groupby("guest_id").size() / weeks
    is_new = first_seen >= NEW_AFTER_SHARE * weeks
    new_rate = data.groupby("guest_id").size()[is_new] / np.maximum(
        weeks - first_seen[is_new], 1)

    segments = pd.Series("occasional", index=first_seen.index)
    segments[rate >= REGULAR_RATE] = "regular"
    segments[rate < LAPSED_RATE] = "lapsed"
    segments[is_new] = "new"
    rate = rate.where(~is_new, new_rate)

    shares, rates = {}, {}
    counts = segments.value_counts()
    for segment in defaults.segment_shares:
        shares[segment] = float(counts.get(segment, 0) / len(segments))
        measured = rate[segments == segment].mean()
        # A segment with no guests, or no visits, keeps the default rate
        rates[segment] = (float(measured) if counts.get(segment, 0) and measured > 0
                          else defaults.segment_visit_rates[segment])
    n_guests = len(first_seen) * np.sqrt(52 / weeks) if weeks < 52 else len(first_seen)
    return n_guests, shares, rates, notes


def _season(transactions: pd.DataFrame, config: PromoConfig, start: pd.Timestamp) -> tuple:
    """Seasonality amplitude, peak week and yearly trend from weekly visits.

    Fits log weekly visits on a straight-line trend plus one yearly cosine, using
    stores that didn't run the promo (or, if every store did, the weeks outside the
    promo and the post window). Only whole weeks are used.
    """
    data = transactions
    controls = ~data["location_id"].isin(config.treated_location_ids)
    if controls.any():
        data = data[controls]
    else:
        outside = (data["timestamp"] < config.promo_start) | (data["timestamp"] >= config.window_end)
        data = data[outside]
    days = (data["timestamp"].dt.normalize() - start).dt.days
    week = days // 7
    visits = week.value_counts().sort_index()
    days_seen = days.groupby(week).nunique()
    visits = visits[days_seen.reindex(visits.index) == 7]
    if len(visits) < MIN_SEASON_WEEKS:
        return None
    t = visits.index.to_numpy(dtype=float)
    angle = 2 * np.pi * t / 52
    design = np.column_stack([np.ones_like(t), t / 52, np.cos(angle), np.sin(angle)])
    coef, *_ = np.linalg.lstsq(design, np.log(visits.to_numpy(dtype=float)), rcond=None)
    amplitude = float(np.clip(np.hypot(coef[2], coef[3]), 0, 0.5))
    peak = int(round(np.degrees(np.arctan2(coef[3], coef[2])) / 360 * 52)) % 52
    trend = float(np.expm1(coef[1]))  # clipped by the caller, after correcting it
    return amplitude, peak, trend


def _corrected_trend(overrides: dict, measured: float) -> float:
    """The simulator's `annual_trend` that reproduces the measured trend.

    New guests arrive through the simulated year, which adds growth on top of
    `annual_trend`. So simulate one year with no trend at all, measure the growth
    the new guests alone add, and divide it out of the measured trend.
    """
    simulated = generate(SynthConfig(seed=0, **{**overrides, "annual_trend": 0.0}))
    season = _season(simulated.transactions, simulated.promo_config,
                     pd.Timestamp(overrides["start_date"]))
    if season is None:
        return float(np.clip(measured, -0.3, 0.3))
    return float(np.clip((1 + measured) / (1 + season[2]) - 1, -0.3, 0.3))


def _segment_weights(shares: dict, rates: dict) -> dict:
    """Each segment's share of all visits."""
    visits = {segment: shares[segment] * rates[segment] for segment in shares}
    total = sum(visits.values()) or 1
    return {segment: value / total for segment, value in visits.items()}


def _past_promo_lifts(transactions, config, shares, rates) -> tuple:
    """Slider values in percent from the file's own promo, or None, plus notes.

    The measured lift during the promo, as a share of the visits promo stores would
    have had anyway, is matched by scaling the default occasional and lapsed lifts
    by one factor (the new-guest lift stays at its default; regulars get none). The
    redeem-anyway share is subsidized redemptions over those expected visits.
    """
    defaults = SynthConfig()
    try:
        effects = did(transactions, config, n_boot=0).set_index("term")
        panel = weekly_panel(transactions, config)
    except Exception:
        return None, ["Your file's promo couldn't be measured, so the lift sliders keep "
                      "their defaults."]
    gross = float(effects.loc["promo_only", "estimate"])
    promo_visits = panel.loc[(panel["treated"] == 1) & (panel["promo"] == 1), "visits"].sum()
    expected = promo_visits - gross
    if not np.isfinite(gross) or gross <= 0 or expected <= 0:
        return None, ["Your file's promo showed no clear lift, so the lift sliders keep "
                      "their defaults."]
    overall = gross / expected

    weights = _segment_weights(shares or defaults.segment_shares,
                               rates or defaults.segment_visit_rates)
    lifts = defaults.lift_by_segment
    fixed = weights["new"] * lifts["new"] + weights["regular"] * lifts["regular"]
    scaled = weights["occasional"] * lifts["occasional"] + weights["lapsed"] * lifts["lapsed"]
    factor = max(overall - fixed, 0) / scaled if scaled > 0 else 1.0

    redemptions = len(promo_redemptions(transactions, config))
    subsidized = float(np.clip(redemptions - gross, 0, redemptions))
    values = {
        "occasional_lift": float(np.clip(100 * lifts["occasional"] * factor, 0, 1000)),
        "lapsed_lift": float(np.clip(100 * lifts["lapsed"] * factor, 0, 1000)),
        "subsidized_share": float(np.clip(100 * subsidized / expected, 0, 100)),
    }
    return values, [f"Lifts set from your past promo: +{overall:.0%} visits during it."]


def calibrate(transactions: pd.DataFrame, config: PromoConfig) -> Calibration:
    """Simulator settings shaped like the business in `transactions`.

    `transactions` and `config` are what `intake.read_upload` returns.
    """
    defaults = SynthConfig()
    notes = []
    start = transactions["timestamp"].min().normalize()
    start = start - pd.Timedelta(days=start.day_of_week)  # the Monday on or before
    market_sizes = _market_sizes(transactions, config)
    n_locations = sum(market_sizes)
    overrides = {
        "market_sizes": market_sizes,
        "start_date": start.strftime("%Y-%m-%d"),
        "n_weeks": 52,
        "treated_location_share": float(np.clip(
            len(config.treated_location_ids) / n_locations, 0.1, 0.9)),
    }
    markets = len(market_sizes)
    notes.append(f"{n_locations} stores in {markets} markets"
                 + ("" if config.location_market else " (no markets in your file, so each "
                    "store counts as its own)") + ".")

    per_store = transactions.groupby("location_id").size()
    if len(per_store) > 1:
        overrides["location_size_sigma"] = float(np.clip(np.log(per_store).std(), 0.05, 1.0))

    n_guests, shares, rates, guest_notes = _guests(transactions, config)
    notes += guest_notes
    if shares:
        overrides["segment_shares"] = shares
        overrides["segment_visit_rates"] = rates
    n_guests = max(int(round(n_guests)), 100)
    if n_guests > MAX_GUESTS:
        notes.append(f"About {n_guests:,} guests a year. The simulation runs at "
                     f"{MAX_GUESTS:,} to stay quick, so its counts are smaller than yours.")
        n_guests = MAX_GUESTS
    else:
        notes.append(f"About {n_guests:,} guests a year.")
    overrides["n_guests"] = n_guests
    if shares:
        notes.append(f"{shares['regular']:.0%} regulars, {shares['occasional']:.0%} occasional, "
                     f"{shares['lapsed']:.0%} lapsed and {shares['new']:.0%} new guests.")

    season = _season(transactions, config, start)
    if season is None:
        notes.append("Less than half a year of history, so seasons follow a typical year.")
    else:
        amplitude, peak, trend = season
        overrides.update(seasonality_amplitude=amplitude, seasonality_peak_week=peak,
                         annual_trend=float(np.clip(trend, -0.3, 0.3)))
        overrides["annual_trend"] = _corrected_trend(overrides, trend)
        busiest = start + pd.Timedelta(weeks=peak)
        # The raw trend includes new guests arriving, so it isn't shown; the simulator
        # uses the corrected trend
        notes.append(f"Busiest around {busiest:%B}, about {amplitude:.0%} above an average week.")

    if _has(transactions, "order_total"):
        totals = pd.to_numeric(transactions["order_total"], errors="coerce")
        if totals.notna().any() and totals.mean() > 0:
            overrides["avg_order_value"] = float(totals.mean())

    redeemed = promo_redemptions(transactions, config)
    discount = (float(redeemed["discount_amount"].mean()) if len(redeemed)
                and redeemed["discount_amount"].mean() > 0 else DEFAULT_DISCOUNT)

    lifts, lift_notes = _past_promo_lifts(transactions, config, shares, rates)
    notes += lift_notes
    # Every key must be a real SynthConfig field
    unknown = set(overrides) - set(defaults.__dataclass_fields__)
    assert not unknown, unknown
    return Calibration(overrides=overrides, margin=float(config.avg_margin_per_order),
                       discount=discount, lifts=lifts, notes=notes)
