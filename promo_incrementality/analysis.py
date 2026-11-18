"""One call from clean transactions to everything a marketer needs to see."""

import dataclasses
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

from promo_incrementality.charts import event_study_chart, weekly_visits_chart
from promo_incrementality.estimators import (
    clusters_by_market,
    PRETREND_FALSE_ALARM_RATE,
    EventStudy,
    did,
    economics,
    event_study,
    headline,
    interval,
    pre_post,
    weekly_panel,
)
from promo_incrementality.intake import MIN_PRE_WEEKS, DataError, check_inputs
from promo_incrementality.power import PowerError, full_weeks, minimum_detectable_effect
from promo_incrementality.schema import PromoConfig

MIN_MARKETS = 10


@dataclass
class Analysis:
    headline: str
    warnings: list
    pre_post: pd.DataFrame
    did: pd.DataFrame
    event_study: EventStudy
    economics: pd.DataFrame
    detectable: dict | None  # from minimum_detectable_effect; None if history is too short
    weekly_chart: go.Figure
    event_study_chart: go.Figure
    ranges: "RangeCoverage"
    config: PromoConfig  # as analyzed: limited to the stores in the data


@dataclass
class RangeCoverage:
    """How often this data's 95% ranges held the truth in testing, as the app labels them."""

    label: str  # e.g. "95%" or "about 90%"
    share: float  # coverage to draw curves with; the nominal 0.95 when testing gives no number
    note: str  # why the label isn't plain 95%; empty when it is


def range_coverage(config: PromoConfig) -> RangeCoverage:
    """Label ranges with the coverage the clustering stress test found for this many markets.

    With 5 markets, ranges built for 95% held the truth about 90% of the time; with
    10 or more, about 95%. Fewer than 5 markets, or no market map, wasn't measured
    at a level the tool can stand behind, so those keep the nominal label with a
    caution.
    """
    markets = config.location_market
    n_markets = len(set(markets.values())) if markets else 0
    if clusters_by_market(markets, markets or []) and n_markets >= MIN_MARKETS:
        return RangeCoverage("95%", 0.95, "")
    if clusters_by_market(markets, markets or []) and n_markets >= 5:
        return RangeCoverage(
            "about 90%", 0.90,
            f"Ranges are built to hold the truth 95% of the time, but with {n_markets} "
            "markets they held it about 90% of the time in testing, so they're labelled 90%.",
        )
    if not markets:
        why = ("with no market map they treat each store as independent, which held the "
               "truth only 72% of the time in testing when a market's stores moved together")
    else:
        why = (f"with only {n_markets} markets, fewer than the 5 tested, they likely hold it "
               "less often than the 90% seen there")
    return RangeCoverage(
        "95% nominal", 0.95,
        f"Ranges are built to hold the truth 95% of the time, but {why}. Treat them as "
        "narrower than labelled.",
    )


def _detectable_effect(transactions: pd.DataFrame, config: PromoConfig) -> dict | None:
    """Run the power calculator on the pre-promo history, if there is enough of it.

    The history starts when the newest location opened, so every location has a
    full baseline. Placebos use a shorter baseline than the real test, to leave
    room for several start weeks; with only one start, the noise estimate would
    rest on market splits alone. On 12 simulated histories this setup's noise
    averaged 624 visits, matching the 624 measured from the actual estimates.
    """
    newest_opening = transactions.groupby("location_id")["timestamp"].min().max().normalize()
    history = transactions[
        (transactions["timestamp"] >= newest_opening)
        & (transactions["timestamp"] < config.promo_start)
    ]
    n_weeks = config.promo_weeks + config.post_window_weeks
    history_weeks = full_weeks(history)
    pre_weeks = max(MIN_PRE_WEEKS, history_weeks - n_weeks - 4)
    if history_weeks < pre_weeks + n_weeks:
        return None

    locations = transactions["location_id"].unique()
    markets = config.location_market or {location: location for location in locations}
    try:
        return minimum_detectable_effect(
            history,
            markets,
            promo_weeks=config.promo_weeks,
            post_window_weeks=config.post_window_weeks,
            pre_weeks=pre_weeks,
            treated_share=len(config.treated_location_ids) / len(locations),
            n_placebos=500,
        )
    except PowerError:  # e.g. stores closed through too much of the history
        return None


def _market_warnings(config: PromoConfig) -> list:
    markets = config.location_market  # already limited to locations in the data
    n_markets = len(set(markets.values())) if markets else 0
    by_market = clusters_by_market(markets, markets or [])
    warnings = []
    if len(config.treated_location_ids) == 1:
        warnings.append(
            "Only one promo location, so ranges rest on a single store and are unreliable."
        )
    if not by_market:
        reason = "No market map was provided" if not markets else f"Only {n_markets} markets"
        warnings.append(
            f"{reason}, so ranges treat each location as independent. They can be much "
            "too narrow if a market's stores move together: in simulation they held only "
            "72% of the time with persistent market shocks."
        )
    elif n_markets < MIN_MARKETS:
        warnings.append(
            f"With only {n_markets} markets, the ranges are a little too narrow: they catch "
            f"the true value about 90% of the time, not 95%. {MIN_MARKETS} or more markets "
            "fix this."
        )
    if not markets:
        return warnings

    treated = set(config.treated_location_ids)
    mixed = sorted(
        {m for location, m in markets.items() if location in treated}
        & {m for location, m in markets.items() if location not in treated}
    )
    if mixed:
        warnings.append(
            f"Markets {mixed} contain both promo and control locations. "
            "Guests can cross over, which usually shrinks the measured effect."
        )
    return warnings


def _warnings(redeemed, config, effects, study, detectable) -> list:
    """Plain-language reasons to be careful with this result."""
    warnings = _market_warnings(config)

    if config.promo_days % 7:
        warnings.append(
            f"The promo lasts {config.promo_days} days, which the analysis rounds up to "
            f"{config.promo_weeks} full weeks, so {7 * config.promo_weeks - config.promo_days} "
            "days after the promo count as promo days."
        )
    if not config.promo_codes:
        codes = redeemed["promo_code"].dropna().unique()
        uncoded = redeemed["promo_code"].isna().sum()
        if len(codes) > 1 or (len(codes) and uncoded):
            kinds = f"{len(codes)} different codes" + (f" and {uncoded:,} with no code" if uncoded else "")
            warnings.append(
                f"Discounts from {kinds} were counted as redemptions. "
                "Enter this promo's codes to count only its redemptions."
            )

    if study.pretrend_p_value < 0.05:
        warnings.append(
            f"Promo and control locations may already have been drifting apart before the "
            f"promo (p = {study.pretrend_p_value:.2f}), which would bias the estimate. Check "
            f"the event study chart. This check false-alarms about "
            f"{PRETREND_FALSE_ALARM_RATE:.0%} of the time."
        )

    net = effects.set_index("term").loc["net"]
    low, high = interval(net)
    if low <= 0 <= high:
        warnings.append("The promo may not have added any visits: zero is inside the range.")
    if detectable and abs(net["estimate"]) < detectable["min_detectable_visits"]:
        warnings.append(
            f"This test is too small to reliably pick up fewer than about "
            f"{detectable['min_detectable_visits']:,.0f} extra visits, and the estimate is "
            f"{net['estimate']:,.0f}. Treat it as a hint, not proof."
        )
    if detectable is None:
        warnings.append("Not enough history to estimate what this test could detect.")
    return warnings


def _limit_to_data(config: PromoConfig, transactions: pd.DataFrame) -> tuple:
    """Leave out stores the market map lists but the data doesn't have.

    A market map often still lists closed stores. Treated and control stores are
    handled alike. A treated store that isn't in the map stays, so check_inputs
    can report it as a likely typo.
    """
    if not config.location_market:
        return config, []
    present = set(transactions["location_id"])
    if not present & set(config.location_market):
        raise DataError([
            "None of the store IDs in the market map match the data. "
            f"The map uses e.g. {sorted(config.location_market)[:3]}, the data "
            f"e.g. {sorted(present)[:3]}."
        ])
    absent = sorted(set(config.location_market) - present)
    treated = tuple(
        l for l in config.treated_location_ids if l in present or l not in config.location_market
    )
    if not treated:
        raise DataError([
            f"None of the promo stores are in the data: {list(config.treated_location_ids)}."
        ])
    config = dataclasses.replace(
        config,
        location_market={l: m for l, m in config.location_market.items() if l in present},
        treated_location_ids=treated,
    )
    note = [f"Stores in the market map but not in the data were left out: {absent}"] if absent else []
    return config, note


def analyze(transactions: pd.DataFrame, config: PromoConfig, n_boot: int = 1999) -> Analysis:
    """Run every estimator on one promo and return results, warnings and charts.

    `n_boot=0` skips the wild bootstrap and falls back to analytic ranges.
    """
    config, left_out = _limit_to_data(config, transactions)
    redeemed = check_inputs(transactions, config)
    panel = weekly_panel(transactions, config)
    effects = did(transactions, config, n_boot=n_boot, panel=panel)
    study = event_study(transactions, config, panel=panel)
    money = economics(transactions, config, effects, redeemed)
    detectable = _detectable_effect(transactions, config)
    return Analysis(
        headline=headline(money),
        warnings=left_out + _warnings(redeemed, config, effects, study, detectable),
        pre_post=pre_post(transactions, config, panel),
        did=effects,
        event_study=study,
        economics=money,
        detectable=detectable,
        weekly_chart=weekly_visits_chart(transactions, config, panel),
        event_study_chart=event_study_chart(study, config),
        ranges=range_coverage(config),
        config=config,
    )
