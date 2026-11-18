"""Bring outside data into the common schema, with errors a person can act on."""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pandas.tseries.api import guess_datetime_format

from promo_incrementality.estimators import promo_redemptions
from promo_incrementality.schema import TRANSACTION_COLUMNS, PromoConfig

REQUIRED_COLUMNS = ["location_id", "timestamp", "discount_amount"]
TEXT_COLUMNS = ["guest_id", "location_id", "promo_code"]
MIN_PRE_WEEKS = 8
CONFIG_KEYS = {
    "promo_start", "promo_end", "post_window_weeks", "avg_margin_per_order",
    "treated_markets", "treated_locations", "markets", "promo_codes", "columns",
    "date_format",
}
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
# A UTC offset after a clock time, e.g. "09:00-05:00", "09:00:00Z", "09:00 +0000", "09:00+05"
UTC_OFFSET = r"^(.*\d:\d\d(?::\d\d(?:\.\d+)?)?)\s?(?:Z|UTC|[+-]\d\d(?::?\d\d)?)$"


def _rows(n: int) -> str:
    return f"{n:,} row has" if n == 1 else f"{n:,} rows have"


BLANKS = {"", "~", "null", "Null", "NULL"}


def _blank_to_none(value):
    """BaseLoader reads a key left blank, or written as null, as text. Make it None."""
    if isinstance(value, dict):
        return {key: _blank_to_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [item for item in map(_blank_to_none, value) if item is not None]
    return None if value in BLANKS else value


def _as_list(value) -> list:
    """YAML lets a one-item list be written as a single value."""
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _as_text(values: pd.Series) -> pd.Series:
    """IDs and codes as text, keeping blanks. Whole floats lose their '.0'."""
    if pd.api.types.is_float_dtype(values) and (values.dropna() % 1 == 0).all():
        values = values.astype("Int64")
    return values.astype("string").astype(object).where(values.notna(), None)


class DataError(ValueError):
    """Input data can't be analyzed. `problems` lists every reason found."""

    def __init__(self, problems: list):
        super().__init__("\n".join(problems))
        self.problems = problems

    def __reduce__(self):
        # Rebuilt from the list, not the joined message, when it crosses processes
        return DataError, (self.problems,)


def _choose_format(text: pd.Series) -> str | None:
    """The single format that reads the most timestamps.

    Candidates come from rows spread across the file, read both month-first and
    day-first. When every date is ambiguous (all days 12 or under), month-first wins.
    """
    samples = text.dropna()
    if samples.empty:
        return None
    if samples.str.match(ISO_DATE).all():
        return "ISO8601"
    spread = samples.iloc[np.linspace(0, len(samples) - 1, min(50, len(samples))).astype(int)]
    with warnings.catch_warnings():  # pandas notes when a guess contradicts dayfirst
        warnings.simplefilter("ignore", UserWarning)
        candidates = [
            guess
            for dayfirst in (False, True)
            for sample in spread
            if (guess := guess_datetime_format(sample, dayfirst=dayfirst))
        ]
    if not candidates:
        return None
    check = samples if len(samples) <= 20_000 else samples.sample(20_000, random_state=0)
    return max(
        dict.fromkeys(candidates),  # unique, month-first guesses first
        key=lambda fmt: pd.to_datetime(check, errors="coerce", format=fmt).notna().sum(),
    )


def _parse_timestamps(values: pd.Series, date_format: str | None) -> pd.Series:
    """Parse timestamps with one format for the whole column.

    Parsing row by row would let "05/07" and "13/07" come out in different orders.
    UTC offsets are dropped, even when `date_format` has %z, so each order keeps its
    local clock time: converting to UTC would move late orders to the next day.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.tz_localize(None) if values.dt.tz is not None else values
    text = values.astype("string").str.strip().str.replace(UTC_OFFSET, r"\1", regex=True)
    if date_format:
        date_format = date_format.replace("%z", "").replace("%Z", "").strip()
    date_format = date_format or _choose_format(text)
    if date_format is None:
        return pd.Series(pd.NaT, index=values.index)
    return pd.to_datetime(text, errors="coerce", format=date_format)


def load_transactions(
    source, column_map: dict | None = None, date_format: str | None = None
) -> pd.DataFrame:
    """Read a CSV path, file object or DataFrame into the common schema.

    `column_map` maps schema names to the source's names, e.g.
    {"location_id": "store", "timestamp": "order_time"}. Only location_id, timestamp
    and discount_amount are required. A blank discount means no discount. IDs are
    read as text, so "0012" stays "0012". `date_format` is a strftime pattern such
    as "%d/%m/%Y %H:%M", needed when the format can't be inferred.
    """
    column_map = column_map or {}
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        text_columns = {column_map.get(column, column): str for column in TEXT_COLUMNS}
        try:
            raw = pd.read_csv(source, dtype=text_columns)
        except pd.errors.EmptyDataError as error:
            raise DataError(["The transactions file is empty."]) from error
        except UnicodeDecodeError as error:
            raise DataError(["The transactions file must be a CSV saved as UTF-8 text."]) from error
        except pd.errors.ParserError as error:
            raise DataError([f"The transactions file isn't a readable CSV: {error}"]) from error

    clashes = [
        f"The file has a '{ours}' column and the config also maps '{theirs}' to it. "
        "Remove one of them."
        for ours, theirs in column_map.items()
        if theirs != ours and ours in raw.columns
    ]
    clashes += [
        f"Unknown column name '{ours}' under columns. Use one of: {', '.join(TRANSACTION_COLUMNS)}."
        for ours in column_map
        if ours not in TRANSACTION_COLUMNS
    ]
    clashes += [
        f"Column '{theirs}' (mapped to {ours}) is not in the file."
        for ours, theirs in column_map.items()
        if ours in TRANSACTION_COLUMNS and ours not in REQUIRED_COLUMNS and theirs not in raw.columns
    ]
    sources = pd.Series(list(column_map.values()))
    clashes += [
        f"'{source}' is mapped to more than one column: "
        f"{', '.join(ours for ours, theirs in column_map.items() if theirs == source)}."
        for source in sources[sources.duplicated()].unique()
    ]
    if clashes:
        raise DataError(clashes)
    data = raw.rename(columns={theirs: ours for ours, theirs in column_map.items()})

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise DataError(
            [
                f"Missing required column '{column_map.get(column, column)}'"
                + (f" (mapped to {column})." if column in column_map else ". Map it in columns.")
                for column in missing
            ]
            + [f"Columns found: {', '.join(map(str, raw.columns))}"]
        )

    if data.empty:
        raise DataError(["The transactions file has column names but no rows."])

    problems = []
    timestamp = _parse_timestamps(data["timestamp"], date_format)
    blank = data["timestamp"].isna() | (data["timestamp"].astype("string").str.strip() == "")
    if blank.any():
        problems.append(f"{_rows(blank.sum())} no timestamp.")
    if (timestamp.isna() & ~blank).any():
        bad = data.loc[timestamp.isna() & ~blank, "timestamp"]
        problems.append(
            f"{_rows(len(bad))} a timestamp that doesn't match the file's format, "
            f"e.g. '{bad.iloc[0]}'. Set date_format in the config, e.g. '%d/%m/%Y %H:%M'."
        )

    blank_discount = data["discount_amount"].isna()
    discount = pd.to_numeric(data["discount_amount"], errors="coerce")
    unreadable = discount.isna() & ~blank_discount
    if unreadable.any():
        example = data.loc[unreadable, "discount_amount"].iloc[0]
        problems.append(f"{_rows(unreadable.sum())} a non-numeric discount, e.g. '{example}'.")
    if (discount < 0).any():
        problems.append(
            f"{_rows((discount < 0).sum())} a negative discount. "
            "Discounts should be positive amounts."
        )

    if data["location_id"].isna().any():
        problems.append(f"{_rows(data['location_id'].isna().sum())} no location_id.")
    if problems:
        raise DataError(problems)

    data = data.assign(timestamp=timestamp, discount_amount=discount.fillna(0.0))
    for column in TRANSACTION_COLUMNS:
        if column not in data.columns:
            data[column] = None
    for column in TEXT_COLUMNS:
        data[column] = _as_text(data[column])
    return data[TRANSACTION_COLUMNS].sort_values("timestamp", ignore_index=True)


def _promo_date(text: str, date_format: str | None) -> pd.Timestamp:
    """A promo date: YYYY-MM-DD, or the date part of the file's `date_format`.

    Any other form is refused rather than guessed, because a day-first date read
    month-first would move the whole promo window. So is a date that YYYY-MM-DD and
    `date_format` would read as two different days.
    """
    readings = []
    if date_format:
        for fmt in (date_format, date_format.split()[0]):
            parsed = pd.to_datetime(text, format=fmt, errors="coerce")
            if pd.notna(parsed):
                readings.append(parsed)
                break
    if ISO_DATE.match(text):
        readings.append(pd.Timestamp(text))
    if not readings:
        raise ValueError(f"write promo dates as YYYY-MM-DD, not '{text}'")
    if len({reading.normalize() for reading in readings}) > 1:
        raise ValueError(
            f"'{text}' reads as {readings[0]:%Y-%m-%d} under date_format but "
            f"{readings[1]:%Y-%m-%d} as YYYY-MM-DD; write it in date_format's style instead"
        )
    return readings[0]


def read_config(source) -> tuple:
    """Read a YAML promo config. Returns (PromoConfig, options for load_transactions).

    Every value is read as text, so store IDs like 0012 keep their zeros. See
    examples/promo_config.yaml. `source` is a path or a file object.
    """
    text = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise DataError(["The config must be saved as UTF-8 text."]) from error
    try:
        raw = yaml.load(text, Loader=yaml.BaseLoader) or {}
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        where = f" near line {mark.line + 1}" if mark else ""
        problem = getattr(error, "problem", None) or "it can't be read"
        raise DataError([f"The config isn't valid YAML{where}: {problem}."]) from error
    if not isinstance(raw, dict):
        raise DataError(["The config should be a set of 'key: value' lines."])
    raw = {key: value for key, value in _blank_to_none(raw).items() if value is not None}

    problems = [f"Unknown config key '{key}'." for key in raw if key not in CONFIG_KEYS]
    problems += [
        f"'{key}' should be 'name: value' lines, not a list."
        for key in ["markets", "columns"]
        if raw.get(key) and not isinstance(raw[key], dict)
    ]
    problems += [
        f"'{key}' should be a single value."
        for key in ["promo_start", "promo_end", "post_window_weeks", "avg_margin_per_order",
                    "date_format"]
        if key in raw and not isinstance(raw[key], str)
    ]
    if isinstance(raw.get("columns"), dict):
        problems += [
            f"Under columns, '{ours}' should name one column."
            for ours, theirs in raw["columns"].items()
            if not isinstance(theirs, str)
        ]
    if problems:
        raise DataError(problems)

    problems += [f"Config is missing '{key}'." for key in ["promo_start", "promo_end",
                                                           "avg_margin_per_order"]
                 if key not in raw]
    if "treated_markets" not in raw and "treated_locations" not in raw:
        problems.append("Config is missing 'treated_markets' or 'treated_locations'.")

    markets = {market: _as_list(locations) for market, locations in (raw.get("markets") or {}).items()}
    location_market = {}
    for market, locations in markets.items():
        for location in locations:
            if location in location_market:
                problems.append(
                    f"Location {location} is listed under both {location_market[location]} "
                    f"and {market}."
                )
            location_market[location] = market
    unknown = [m for m in _as_list(raw.get("treated_markets")) if m not in markets]
    if unknown:
        problems.append(f"treated_markets not listed under markets: {unknown}")
    if problems:
        raise DataError(problems)

    numbers = {
        "avg_margin_per_order": (float, "a dollar amount, like 6.0"),
        "post_window_weeks": (int, "a whole number of weeks, like 6"),
    }
    for key, (kind, example) in numbers.items():
        try:
            if key in raw:
                kind(raw[key])
        except ValueError:
            problems.append(f"'{key}' should be {example}, not '{raw[key]}'.")
    if problems:
        raise DataError(problems)

    treated = _as_list(raw.get("treated_locations"))
    for market in _as_list(raw.get("treated_markets")):
        treated += markets[market]
    if not treated:
        raise DataError(["No promo locations: treated_markets and treated_locations are empty."])
    try:
        config = PromoConfig(
            promo_start=_promo_date(raw["promo_start"], raw.get("date_format")),
            promo_end=_promo_date(raw["promo_end"], raw.get("date_format")),
            treated_location_ids=sorted(set(treated)),
            post_window_weeks=int(raw.get("post_window_weeks", 6)),
            avg_margin_per_order=float(raw["avg_margin_per_order"]),
            location_market=location_market or None,
            promo_codes=_as_list(raw.get("promo_codes")) or None,
        )
    except (ValueError, TypeError) as error:
        raise DataError([f"Bad value in the config: {error}"]) from error
    options = {"column_map": raw.get("columns") or {}, "date_format": raw.get("date_format")}
    return config, options


def check_inputs(transactions: pd.DataFrame, config: PromoConfig) -> pd.DataFrame:
    """Raise DataError if this data can't support an analysis of this promo.

    Returns the promo's redemptions, which the check has to find anyway.
    """
    problems = []
    locations = set(transactions["location_id"])
    treated = set(config.treated_location_ids)

    if treated and not treated & locations:
        problems.append("None of the treated locations appear in the data.")
    elif treated - locations:
        problems.append(f"Treated locations not in the data: {sorted(treated - locations)}")
    if not locations - treated:
        problems.append("Every location is treated, so there is no control group.")
    if len(locations) < 3:
        problems.append("At least 3 locations are needed.")
    if config.location_market and locations - set(config.location_market):
        missing = sorted(locations - set(config.location_market))
        problems.append(f"Locations missing from the market map: {missing}")
    if not np.isfinite(config.avg_margin_per_order) or config.avg_margin_per_order <= 0:
        problems.append("avg_margin_per_order must be above zero.")
    if config.post_window_weeks < 1:
        problems.append(
            "post_window_weeks must be at least 1, so pull-forward and halo can be measured."
        )

    needed = max(MIN_PRE_WEEKS, config.promo_weeks)
    first_day = transactions.groupby("location_id")["timestamp"].min().dt.normalize()
    history_weeks = (config.promo_start - first_day).dt.days // 7
    before = transactions[transactions["timestamp"] < config.promo_start]
    week = (before["timestamp"].dt.normalize() - config.promo_start).dt.days // 7
    active_weeks = week.groupby(before["location_id"]).nunique().reindex(history_weeks.index, fill_value=0)
    sparse = sorted(set(active_weeks[active_weeks < needed].index) - set(history_weeks[history_weeks < needed].index))
    if sparse:
        problems.append(
            f"Locations with visits in fewer than {needed} weeks before the promo: {sparse}. "
            "A stray early order doesn't make a baseline; remove them."
        )
    short = sorted(history_weeks[history_weeks < needed].index)
    if short:
        problems.append(
            f"Locations with fewer than {needed} full weeks of data before the promo: "
            f"{short}. Remove them, or analyze a later promo."
        )

    last_needed = config.last_day_needed
    if transactions["timestamp"].max().normalize() < last_needed:
        problems.append(
            f"Data ends {transactions['timestamp'].max():%Y-%m-%d}, before the post window "
            f"ends {last_needed:%Y-%m-%d}. Shorten post_window_weeks or add data."
        )
    else:
        last_day = transactions.groupby("location_id")["timestamp"].max().dt.normalize()
        closed = sorted(last_day[last_day < last_needed - pd.Timedelta(days=6)].index)
        if closed:
            problems.append(
                f"Locations with no visits in the last week of the post window: {closed}. "
                "If they closed, remove them."
            )

    redeemed = promo_redemptions(transactions, config)
    if not problems and redeemed.empty:
        problems.append(
            "No discounted orders at promo locations during the promo. Check the discount "
            "column mapping and promo_codes."
        )
    if problems:
        raise DataError(problems)
    return redeemed


# The one-file upload: transactions with each store's market and promo status as columns,
# so no config file is needed. The promo dates and margin are entered on the page.
UPLOAD_REQUIRED = ["location_id", "got_promo", "timestamp", "discount_amount"]
UPLOAD_COLUMNS = UPLOAD_REQUIRED + ["market", "promo_code", "guest_id", "order_total", "channel"]
YES = {"yes", "y", "true", "1"}
NO = {"no", "n", "false", "0"}


def _per_store(data: pd.DataFrame, column: str) -> tuple:
    """Each store's single value in `column`, and problems for stores with more than one."""
    values = data.groupby("location_id")[column].unique()
    mixed = values[values.map(len) > 1]
    problems = [
        f"Store {store} has more than one {column}: {', '.join(map(str, sorted(found)))}. "
        "Each store needs one value on every row."
        for store, found in mixed.head(5).items()
    ]
    return values.map(lambda found: found[0]), problems


def read_upload(
    source,
    promo_start,
    promo_end,
    avg_margin_per_order: float,
    post_window_weeks: int = 6,
    promo_codes: list | None = None,
) -> tuple:
    """Read the one-file upload. Returns (transactions, PromoConfig).

    Columns use the names in UPLOAD_COLUMNS. `got_promo` (yes/no) marks the stores
    that ran the promo, and `market`, if given, groups stores into markets.
    """
    text_columns = {column: str for column in ["location_id", "market", "got_promo",
                                                "promo_code", "guest_id"]}
    try:
        raw = pd.read_csv(source, dtype=text_columns)
    except pd.errors.EmptyDataError as error:
        raise DataError(["The file is empty."]) from error
    except UnicodeDecodeError as error:
        raise DataError(["The file must be a CSV saved as UTF-8 text."]) from error
    except pd.errors.ParserError as error:
        raise DataError([f"The file isn't a readable CSV: {error}"]) from error
    raw.columns = [str(column).strip().lower() for column in raw.columns]

    missing = [column for column in UPLOAD_REQUIRED if column not in raw.columns]
    if missing:
        raise DataError(
            [f"Missing required column '{column}'." for column in missing]
            + [f"Columns found: {', '.join(raw.columns)}",
               "Use the AI instructions on this page to rename your columns."]
        )
    try:
        transactions = load_transactions(raw[[c for c in raw.columns if c in TRANSACTION_COLUMNS]])
    except DataError as error:
        # The loader's advice points at the config, which this upload doesn't have
        raise DataError([
            problem.replace("Set date_format in the config, e.g. '%d/%m/%Y %H:%M'.",
                            "Write every timestamp as YYYY-MM-DD HH:MM:SS.")
            for problem in error.problems
        ]) from error

    status = raw["got_promo"].astype("string").str.strip().str.lower()
    unreadable = status.isna() | ~status.isin(YES | NO)
    problems = []
    if unreadable.any():
        example = raw.loc[unreadable, "got_promo"].iloc[0]
        problems.append(f"{_rows(unreadable.sum())} a got_promo value that isn't yes or no, "
                        f"e.g. '{example}'.")
    has_market = "market" in raw.columns and raw["market"].notna().any()
    if has_market and raw["market"].isna().any():
        problems.append(f"{_rows(raw['market'].isna().sum())} no market. Fill in every row's "
                        "market, or leave the whole column out.")
    if problems:
        raise DataError(problems)

    data = pd.DataFrame({
        "location_id": _as_text(raw["location_id"]),
        "got_promo": status.isin(YES),
    })
    if has_market:
        data["market"] = _as_text(raw["market"])
    promo, problems = _per_store(data, "got_promo")
    location_market = None
    if has_market:
        markets, more = _per_store(data, "market")
        problems += more
        location_market = markets.to_dict()
    if problems:
        raise DataError(problems)
    treated = sorted(promo[promo].index)
    if not treated:
        raise DataError(["No store has got_promo = yes, so there's no promo to measure."])

    try:
        config = PromoConfig(
            promo_start=pd.Timestamp(promo_start),
            promo_end=pd.Timestamp(promo_end),
            treated_location_ids=treated,
            post_window_weeks=int(post_window_weeks),
            avg_margin_per_order=float(avg_margin_per_order),
            location_market=location_market,
            promo_codes=list(promo_codes) if promo_codes else None,
        )
    except ValueError as error:
        raise DataError([f"Check the promo settings: {error}."]) from error
    return transactions, config


def to_upload(transactions: pd.DataFrame, config: PromoConfig) -> pd.DataFrame:
    """Transactions in the one-file upload format, with market and got_promo filled in
    from `config`. Builds the example file from the sample data and config."""
    treated = set(config.treated_location_ids)
    upload = transactions.assign(
        market=transactions["location_id"].map(config.location_market or {}),
        got_promo=transactions["location_id"].isin(treated).map({True: "yes", False: "no"}),
        timestamp=transactions["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
    )
    return upload[[column for column in UPLOAD_COLUMNS if upload[column].notna().any()]]
