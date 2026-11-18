"""The common schema: every data source maps to these columns and this config."""

from dataclasses import dataclass

import pandas as pd

TRANSACTION_COLUMNS = [
    "guest_id",
    "location_id",
    "timestamp",
    "order_total",
    "discount_amount",
    "promo_code",
    "channel",
]


@dataclass(frozen=True)
class PromoConfig:
    """What a marketer knows about a promo. `promo_end` is the last promo day.

    `location_market` maps every location_id to its market. When provided, standard
    errors are clustered by market, the level the promo was assigned at.
    `promo_codes` limits redemptions to those codes. Without it, every discounted
    order at a promo location during the promo counts as a redemption.
    """

    promo_start: pd.Timestamp
    promo_end: pd.Timestamp
    treated_location_ids: tuple
    post_window_weeks: int = 6
    avg_margin_per_order: float = 6.0
    location_market: dict | None = None
    promo_codes: tuple | None = None

    def __post_init__(self):
        # Whole days: weeks and the promo window are counted in calendar days.
        object.__setattr__(self, "promo_start", pd.Timestamp(self.promo_start).normalize())
        object.__setattr__(self, "promo_end", pd.Timestamp(self.promo_end).normalize())
        object.__setattr__(self, "treated_location_ids", tuple(self.treated_location_ids))
        if self.promo_codes is not None:
            object.__setattr__(self, "promo_codes", tuple(self.promo_codes))
        if self.promo_end < self.promo_start:
            raise ValueError("promo_end is before promo_start")

    @property
    def promo_days(self) -> int:
        return (self.promo_end - self.promo_start).days + 1

    @property
    def promo_weeks(self) -> int:
        """Promo length in whole weeks, rounded up."""
        return -(-self.promo_days // 7)

    @property
    def window_end(self) -> pd.Timestamp:
        """The day after the post window ends."""
        return self.promo_start + pd.Timedelta(weeks=self.promo_weeks + self.post_window_weeks)

    @property
    def last_day_needed(self) -> pd.Timestamp:
        """Data must reach this day, the last of the post window."""
        return self.window_end - pd.Timedelta(days=1)


def validate_transactions(df: pd.DataFrame) -> None:
    """Raise if `df` does not follow the common schema."""
    missing = set(TRANSACTION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"transactions are missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError("timestamp must be a datetime column")
