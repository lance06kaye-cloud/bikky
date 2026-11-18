"""Map the Starbucks Rewards offers dataset to the common schema.

The data comes from Udacity's data science capstone and is mirrored on Kaggle
(blacktile/starbucks-app-customer-reward-program-data). Udacity describes it as
simulated data that mimics customer behavior on the rewards app, so it shows the
tool working on an outside dataset, not real restaurant results.

Assumptions:

Schema mapping
- guest_id is `person`. Each "transaction" event is one visit.
- location_id is the single value "app". The data has no stores, which is why the
  location-based estimators (`pre_post`, `did`, `event_study`) don't apply here.
- timestamp is `time` in hours after START_DATE. The start date is arbitrary.
- order_total is the transaction `amount`.
- discount_amount is the summed `reward` of every offer completed by that
  transaction. A completion always shares its person and hour with a transaction,
  and rewards are treated as a discount even though they are paid out as points.
- promo_code is the completed offer's id. One purchase can complete several offers,
  so several ids are joined with "|", sorted. It is empty for no completed offer.
- channel is the completed offers' portfolio channels, joined with "|". Those are
  the channels the offer was sent through, not where the purchase happened.
  Transactions that completed no offer get "app".

Offer attribution
- An offer copy is valid from receipt through receipt + `duration` days.
- Customers can hold overlapping copies of the same offer, and one purchase can
  complete both. That produces the identical "offer completed" rows in the raw
  file (397 of them). They are real rewards and are kept, not de-duplicated.
- Each completion goes to the earliest valid copy not yet completed.
- A view counts for a copy if it falls inside that copy's valid window, so one view
  can count for two overlapping copies. That makes the unviewed counts conservative.
- A view in the same hour as the completion counts as before it. Hours are the
  finest resolution in the data, so this also keeps the counts conservative.
- `never_viewed` copies were completed with no view in the valid window.
  `viewed_after_completion` copies were seen only after the reward was earned. In
  both cases the offer could not have influenced the purchase, so the reward is
  pure subsidy. Informational offers have no reward and are never completed.

Age 118 in profile.json is a placeholder for missing demographics. Those customers
stay in, because the analyses here don't use demographics.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from promo_incrementality.schema import TRANSACTION_COLUMNS

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_FILES = ["portfolio.json", "profile.json", "transcript.json"]
START_DATE = pd.Timestamp("2018-01-01")
LOCATION_ID = "app"


@dataclass
class StarbucksData:
    transactions: pd.DataFrame  # common schema
    offers: pd.DataFrame  # one row per offer copy received
    guest_ids: pd.Index  # every customer, including any with no offers
    subsidy: dict  # completed-without-viewing counts


def load_starbucks(raw_dir: Path | str = DEFAULT_RAW_DIR) -> StarbucksData:
    """Read the raw files and return common-schema transactions plus offer detail."""
    raw_dir = Path(raw_dir)
    missing = [name for name in RAW_FILES if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {raw_dir}. Download the Starbucks rewards dataset "
            "from Kaggle and put the three JSON files there."
        )
    portfolio = pd.read_json(raw_dir / "portfolio.json", lines=True)
    profile = pd.read_json(raw_dir / "profile.json", lines=True)
    events = _flatten_events(pd.read_json(raw_dir / "transcript.json", lines=True))

    offers = _build_offers(events, portfolio)
    transactions = _build_transactions(events, offers)
    return StarbucksData(
        transactions=transactions,
        offers=offers,
        guest_ids=pd.Index(profile["id"], name="guest_id"),
        subsidy=_summarize_subsidy(offers),
    )


def _flatten_events(transcript: pd.DataFrame) -> pd.DataFrame:
    """Unpack the `value` dicts. The raw file spells the key two ways."""
    value = transcript["value"]
    return pd.DataFrame(
        {
            "guest_id": transcript["person"],
            "event": transcript["event"],
            "hour": transcript["time"],
            "offer_id": value.map(lambda v: v.get("offer id", v.get("offer_id"))),
            "amount": value.map(lambda v: v.get("amount")),
            "reward": value.map(lambda v: v.get("reward")),
        }
    )


def _build_offers(events: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    """Build one row per offer copy, with its completion and view status."""
    details = portfolio.rename(columns={"id": "offer_id"})
    details["channel"] = details["channels"].map(lambda c: "|".join(sorted(c)))
    offers = (
        events.loc[events["event"] == "offer received", ["guest_id", "offer_id", "hour"]]
        .rename(columns={"hour": "received_hour"})
        .merge(
            details[["offer_id", "offer_type", "reward", "duration", "channel"]], on="offer_id"
        )
        .sort_values("received_hour", ignore_index=True)
    )
    offers["expires_hour"] = offers["received_hour"] + offers["duration"] * 24
    offers["completed_hour"] = _match_completions(offers, events)

    views = events.loc[events["event"] == "offer viewed", ["guest_id", "offer_id", "hour"]]
    candidates = (
        offers[offers["completed_hour"].notna()]
        .reset_index()
        .merge(views, on=["guest_id", "offer_id"])
    )
    in_window = candidates["hour"].between(
        candidates["received_hour"], candidates["expires_hour"]
    )
    before = in_window & (candidates["hour"] <= candidates["completed_hour"])
    viewed_before = candidates.loc[before, "index"]
    viewed_any = candidates.loc[in_window, "index"]

    completed = offers["completed_hour"].notna()
    offers["viewed_before_completion"] = offers.index.isin(viewed_before)
    offers["viewed_after_completion"] = offers.index.isin(viewed_any) & ~offers.index.isin(
        viewed_before
    )
    offers["never_viewed"] = completed & ~offers.index.isin(viewed_any)
    return offers


def _match_completions(offers: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    """Give each completion to the earliest valid copy that is still open."""
    copies = offers.groupby(["guest_id", "offer_id"], sort=False).indices
    received, expires = offers["received_hour"].to_numpy(), offers["expires_hour"].to_numpy()
    completed_hour = np.full(len(offers), np.nan)

    completions = events[events["event"] == "offer completed"].sort_values("hour")
    for guest_id, offer_id, hour in completions[["guest_id", "offer_id", "hour"]].itertuples(
        index=False
    ):
        for copy in copies.get((guest_id, offer_id), []):
            is_open = np.isnan(completed_hour[copy])
            if is_open and received[copy] <= hour <= expires[copy]:
                completed_hour[copy] = hour
                break
        else:
            raise ValueError(f"no valid offer copy for completion by {guest_id} at {hour}")
    return completed_hour


def _build_transactions(events: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    """One row per purchase, carrying the rewards of any offers it completed."""
    purchases = events.loc[events["event"] == "transaction", ["guest_id", "hour", "amount"]]

    def join_unique(values):
        return "|".join(sorted(set(values)))

    def join_channels(values):
        return join_unique(c for channels in values for c in channels.split("|"))

    rewards = (
        offers[offers["completed_hour"].notna()]
        .groupby(["guest_id", "completed_hour"], as_index=False)
        .agg(
            discount_amount=("reward", "sum"),
            promo_code=("offer_id", join_unique),
            offer_channel=("channel", join_channels),
        )
        .rename(columns={"completed_hour": "hour"})
    )
    merged = purchases.merge(rewards, on=["guest_id", "hour"], how="left")
    # Two purchases in the same hour: give the reward to the first only.
    repeat = merged.duplicated(["guest_id", "hour"])
    merged.loc[repeat, ["discount_amount", "promo_code", "offer_channel"]] = np.nan
    if merged["discount_amount"].sum() != rewards["discount_amount"].sum():
        raise ValueError("some completed offers have no matching transaction")

    return (
        merged.assign(
            location_id=LOCATION_ID,
            timestamp=START_DATE + pd.to_timedelta(merged["hour"], unit="h"),
            order_total=merged["amount"],
            discount_amount=merged["discount_amount"].fillna(0.0),
            channel=merged["offer_channel"].fillna(LOCATION_ID),
        )[TRANSACTION_COLUMNS]
        .sort_values("timestamp", ignore_index=True)
    )


def _summarize_subsidy(offers: pd.DataFrame) -> dict:
    """Count completions the offer could not have caused, and what they cost."""
    completed = offers[offers["completed_hour"].notna()]
    never = completed[completed["never_viewed"]]
    after = completed[completed["viewed_after_completion"]]
    unviewed = len(never) + len(after)
    return {
        "completed_offers": len(completed),
        "reward_cost": int(completed["reward"].sum()),
        "completed_never_viewed": len(never),
        "never_viewed_reward_cost": int(never["reward"].sum()),
        "completed_before_viewing": len(after),
        "before_viewing_reward_cost": int(after["reward"].sum()),
        "unviewed_share_of_completions": unviewed / len(completed),
        "unviewed_share_of_reward_cost": float(
            (never["reward"].sum() + after["reward"].sum()) / completed["reward"].sum()
        ),
    }


def wave_comparison(data: StarbucksData) -> pd.DataFrame:
    """Compare first-time offer recipients with customers not yet sent an offer.

    Offers go out in waves. For each wave, recipients are customers getting their
    first offer in it, and the not-yet group has received nothing so far. Visits and
    spend are measured per customer from that wave to the next, so the not-yet group
    is untouched throughout.

    `visit_effect` and `spend_effect` are differences in the change from the
    same-length window before the wave, a customer-level DiD. Wave 0 has no earlier
    window and uses the plain difference. `pre_visit_gap` is the recipient minus
    not-yet gap before the wave: near zero means the groups started out alike.
    """
    first_offer = (
        data.offers.groupby("guest_id")["received_hour"].min().reindex(data.guest_ids)
    )
    purchases = data.transactions.assign(
        hour=(data.transactions["timestamp"] - START_DATE) / pd.Timedelta(hours=1)
    )
    waves = sorted(data.offers["received_hour"].unique())
    window_ends = waves[1:] + [purchases["hour"].max() + 1]

    def per_guest(start, end):
        window = purchases[purchases["hour"].between(start, end, inclusive="left")]
        by_guest = window.groupby("guest_id")["order_total"]
        return (
            by_guest.size().reindex(data.guest_ids, fill_value=0),
            by_guest.sum().reindex(data.guest_ids, fill_value=0.0),
        )

    rows = []
    for wave, end in zip(waves, window_ends):
        recipients = first_offer == wave
        not_yet = ~(first_offer <= wave)
        visits, spend = per_guest(wave, end)
        has_pre_window = wave - (end - wave) >= 0
        pre_gap = np.nan
        if has_pre_window:
            pre_visits, pre_spend = per_guest(wave - (end - wave), wave)
            pre_gap = pre_visits[recipients].mean() - pre_visits[not_yet].mean()
            visits, spend = visits - pre_visits, spend - pre_spend

        test = stats.ttest_ind(visits[recipients], visits[not_yet], equal_var=False)
        interval = test.confidence_interval(0.95)
        rows.append(
            {
                "wave_hour": wave,
                "window_hours": end - wave,
                "n_recipients": recipients.sum(),
                "n_not_yet": not_yet.sum(),
                "pre_visit_gap": pre_gap,
                "visit_effect": visits[recipients].mean() - visits[not_yet].mean(),
                "ci_low": interval.low,
                "ci_high": interval.high,
                "spend_effect": spend[recipients].mean() - spend[not_yet].mean(),
                "method": "did" if has_pre_window else "difference",
            }
        )
    return pd.DataFrame(rows)
