"""How numbers read in the headline, tables and app, so they always match."""

import pandas as pd


def visits_text(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:,.0f}"


def money_text(value: float) -> str:
    """Whole dollars, e.g. -$1,250."""
    if pd.isna(value):
        return "n/a"
    value = round(value)
    return f"{'-' if value < 0 else ''}${abs(value):,.0f}"


def dollars_text(value: float) -> str:
    """Dollars and cents under $100, e.g. -$5.25, and whole dollars above."""
    if pd.isna(value) or abs(value) >= 100:
        return money_text(value)
    value = round(value, 2)
    return f"{'-' if value < 0 else ''}${abs(value):,.2f}"
