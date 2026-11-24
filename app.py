"""Promo incrementality demo. Run with: streamlit run app.py"""

import html
import io
import json
import secrets
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from promo_incrementality.analysis import analyze
from promo_incrementality.calibrate import MAX_GUESTS, calibrate
from promo_incrementality.charts import (
    chance_above,
    density_chart,
    density_from_interval,
    density_from_samples,
    event_study_chart,
    weekly_means_chart,
)
from promo_incrementality.estimators import interval
from promo_incrementality.formatting import dollars_text, money_text, visit_cost_text, visits_text
from promo_incrementality.intake import (
    MIN_PRE_WEEKS,
    DataError,
    check_settings,
    load_transactions,
    read_config,
    read_upload,
    to_upload,
)
from promo_incrementality.load_starbucks import load_starbucks, wave_comparison
from promo_incrementality.montecarlo import IN_BROWSER, run_worlds
from promo_incrementality.synth import SynthConfig
from promo_incrementality.theme import FAINT_TEXT, css

# The browser runs one simulated promo at a time, so it runs fewer
N_WORLDS = 25 if IN_BROWSER else 100
EXAMPLES = Path(__file__).parent / "examples"


def middle_95(values) -> tuple:
    return tuple(np.nanquantile(values, [0.025, 0.975]))


def view_key(name: str) -> str:
    """A widget key that changes on reset, which clears zoom and selections."""
    return f"{name}_{st.session_state.view}"


def show_chart(figure, name: str, **options):
    return st.plotly_chart(figure, width="stretch", key=view_key(name), **options)


# Cached work


# The discount the lifts are typed for. Discount elasticity scales lifts relative to it.
REFERENCE_DISCOUNT = 1.75


def discount_scale(discount: float, elasticity: float, reference: float) -> float:
    """How much the lifts grow with the discount: (discount / reference) ^ elasticity.

    Elasticity 0 leaves the lifts as typed, whatever the discount.
    """
    return (discount / reference) ** elasticity if elasticity else 1.0


def planted_effects(
    discount, margin, occasional_lift, lapsed_lift, new_lift, elasticity, subsidized_share,
    n_guests=SynthConfig().n_guests, start_week=24, promo_weeks=4, reference=REFERENCE_DISCOUNT,
) -> dict:
    scale = discount_scale(discount, elasticity, reference)
    return {
        "promo_start_week": start_week,
        "promo_weeks": promo_weeks,
        "discount_amount": discount,
        "avg_margin_per_order": margin,
        # Regulars have no lift; every other segment has its own sidebar input
        "lift_by_segment": {
            **SynthConfig().lift_by_segment,
            "occasional": occasional_lift * scale,
            "lapsed": lapsed_lift * scale,
            "new": new_lift * scale,
        },
        "subsidized_share": subsidized_share,
        "n_guests": int(n_guests),
    }


# Cached as a shared resource, not copied on every rerun. The result is never changed.
@st.cache_resource(show_spinner=f"Simulating {N_WORLDS} promos...", max_entries=8)
def simulate(first_seed, learned: str, *settings):
    """`learned` is SynthConfig overrides learned from an upload, as JSON ("{}" for none).

    JSON keeps the cache key hashable. The sidebar's planted effects win over it.
    """
    # The seed comes from first_seed, and the timeline is fitted to a 52-week year
    overrides = {key: tuple(value) if isinstance(value, list) else value
                 for key, value in json.loads(learned).items()
                 if key not in ("seed", "n_weeks")}
    return run_worlds(N_WORLDS, first_seed, analyze_first=True,
                      **{**overrides, **planted_effects(*settings)})


def as_json(overrides: dict) -> str:
    """Overrides as sorted JSON, with numpy numbers turned into plain ones."""
    return json.dumps(overrides, sort_keys=True,
                      default=lambda value: value.item() if hasattr(value, "item") else list(value))


@st.cache_resource(show_spinner=False, max_entries=4)
def analyze_upload(csv_bytes: bytes, *settings):
    """Analyze the one-file upload. `settings` are read_upload's promo settings."""
    return analyze(*read_upload(io.BytesIO(csv_bytes), *settings))


@st.cache_resource(show_spinner=False, max_entries=4)
def calibrate_upload(csv_bytes: bytes, *settings):
    """What the upload says about the business, for simulating it. `settings` as above."""
    return calibrate(*read_upload(io.BytesIO(csv_bytes), *settings))


@st.cache_resource(show_spinner="Preparing the example file...")
def example_upload() -> tuple:
    """The example file in the upload format, and the promo settings it goes with.

    Built from the sample CSV and config, so there's one source for the example.
    """
    config, options = read_config(EXAMPLES / "promo_config.yaml")
    transactions = load_transactions(EXAMPLES / "sample_transactions.csv", **options)
    settings = {
        "promo_start": config.promo_start.date(),
        "promo_end": config.promo_end.date(),
        "margin": config.avg_margin_per_order,
        "post_weeks": config.post_window_weeks,
        "promo_codes": ", ".join(config.promo_codes or []),
    }
    return to_upload(transactions, config).to_csv(index=False).encode(), settings


@st.cache_data(show_spinner="Loading Starbucks data...")
def run_starbucks():
    data = load_starbucks()
    return data.subsidy, wave_comparison(data)


# Simulated tab


# Label, truth column, estimate column, and how to format the numbers
METRICS = [
    ("Net incremental visits", "true_net", "did_net", visits_text),
    ("Lift during the promo", "true_promo", "did_promo", visits_text),
    ("After the promo (halo minus pull-forward)", "true_post", "did_post", visits_text),
    ("Subsidized redemptions", "true_subsidized", "subsidized", visits_text),
    ("Cost per net incremental visit", "true_cost_per_visit", "cost_per_visit", visit_cost_text),
    ("Net profit", "true_profit", "net_profit", money_text),
]
# Every number on the simulated view is either planted (true) or estimated, and says which
TRUE_COLUMN, ESTIMATE_COLUMN = "True (planted)", "Estimated"
TRUE_MARKER = "True (planted) average"
# The same rounding as each formatter above, for chart axes and hover (d3 formats)
CHART_FORMATS = {visits_text: ",.0f", visit_cost_text: "$,.2f", money_text: "$,.0f"}


def average(runs: pd.DataFrame, column: str) -> float:
    """Average a column across runs.

    Cost per visit is total cost over total visits. Averaging each run's ratio would
    let runs with almost no net visits dominate, and skip runs with none.
    """
    if column in ("cost_per_visit", "true_cost_per_visit"):
        visits = runs["did_net" if column == "cost_per_visit" else "true_net"].mean()
        return runs["discount_cost"].mean() / visits if visits > 0 else np.nan
    return runs[column].mean()


def not_meaningful(runs: pd.DataFrame) -> int:
    """Runs whose estimated net visits are zero or below, so cost per visit means nothing."""
    return int((runs["did_net"] <= 0).sum())


def metrics_table(runs: pd.DataFrame) -> pd.DataFrame:
    """True (planted) average, estimated average and middle 95% for each METRICS row.

    Cost per visit's range leaves out runs with no estimated net visits: a cost
    divided by zero or fewer visits is undefined, not a large number.
    """
    rows = []
    for label, truth, estimate, text in METRICS:
        values = runs[estimate]
        spread = "n/a"
        if estimate == "cost_per_visit":
            values = values[runs["did_net"] > 0]
            spread = "No run had net visits"
        if values.notna().any():
            low, high = middle_95(values)
            spread = f"{text(low)} to {text(high)}"
        if estimate == "cost_per_visit" and not_meaningful(runs):
            spread += f" ({not_meaningful(runs)} runs left out)"
        rows.append(
            {
                "Average per promo": label,
                TRUE_COLUMN: text(average(runs, truth)),
                ESTIMATE_COLUMN: text(average(runs, estimate)),
                "Middle 95% of runs": spread,
            }
        )
    return pd.DataFrame(rows)


def metric_distribution_chart(runs: pd.DataFrame, row: int):
    """Probability density of one METRICS row across the simulated promos.

    Raises ValueError when the runs don't vary enough to draw a curve.
    """
    label, truth, estimate, text = METRICS[row]
    markers = [(average(runs, truth), TRUE_MARKER)]
    if estimate == "did_net":
        markers.append((runs["break_even_visits"].mean(), "Break-even"))
    elif estimate == "net_profit":
        markers.append((0.0, "Break-even"))
    elif estimate == "cost_per_visit":
        markers.append((runs["cost_per_redemption"].mean(), "Face value"))
    x, density = density_from_samples(runs[estimate])
    title = f"{label}: estimates across {len(runs)} simulated promos"
    return density_chart(x, density, title, label, markers, CHART_FORMATS[text])


def show_clicked_week(selection, draw, key: str):
    """Show the promo effect's distribution in the week the viewer clicked.

    `draw(week)` returns (x, density, title), or raises ValueError with a reason to
    show instead.
    """
    points = selection.selection.points
    if not points:
        st.caption("Click any dot to see the distribution for that week.")
        return
    week = int(points[0]["x"])
    if week == -1:
        st.info("Week -1 is the comparison base, so it has no distribution.")
        return
    try:
        x, density, title = draw(week)
    except ValueError as reason:
        st.info(str(reason))
        return
    chart = density_chart(x, density, title, "Visits per location vs. week -1", [(0.0, "No effect")])
    show_chart(chart, f"{key}_week")


def simulated_week(weekly_gaps: pd.DataFrame):
    def draw(week):
        gaps = weekly_gaps.loc[weekly_gaps["week"] == week, "estimate"]
        try:
            x, density = density_from_samples(gaps)
        except ValueError:
            raise ValueError(f"Week {week} came out the same in every run, so there's no distribution.")
        return x, density, f"Week {week}: promo effect across {len(gaps)} simulated promos"

    return draw


def estimated_week(coefficients: pd.DataFrame):
    def draw(week):
        row = coefficients.set_index("week").loc[week]
        if not (np.isfinite([row["ci_low"], row["ci_high"]]).all() and row["ci_high"] > row["ci_low"]):
            raise ValueError(f"Week {week} has no range to draw.")
        x, density = density_from_interval(row["estimate"], row["ci_low"], row["ci_high"])
        return x, density, f"Week {week}: likely promo effect (normal approximation)"

    return draw


def show_headline(key: str, detail: str):
    """A headline styled like a heading, but paragraphs: the key result large, then
    the sentence behind it.

    Streamlit gives every heading an id built from its text, even with anchor=False,
    so a heading with numbers would leave stale numbers in a copied link.
    """
    st.html(f'<p class="headline-key">{html.escape(key)}</p>'
            f'<p class="headline-detail">{html.escape(detail)}</p>')


def nearest_ten(value: float) -> str:
    """Visits to the nearest 10, rounded before formatting so it never reads "-0"."""
    return f"{round(value, -1) + 0.0:,.0f}"


def signed_ten(value: float) -> str:
    """Visits to the nearest 10 with a sign, e.g. +230 or -870, and plain 0 for zero."""
    rounded = round(value, -1)
    return "0" if rounded == 0 else f"{rounded:+,.0f}"


def whole_percent(share: float) -> str:
    """A whole percentage, with "<1%" so a small nonzero change doesn't read as 0%."""
    return "<1%" if 0 < share < 0.005 else f"{share:.0%}"


def share_of(change: float, base: float) -> str | None:
    """The change as a whole percentage of base, or None when base is zero or less."""
    return whole_percent(abs(change) / base) if base > 0 else None


def break_even_levers(net, redemptions, subsidized, discount, margin) -> dict:
    """What, changed alone, would bring the promo's profit to zero.

    Profit = net visits × margin − redemptions × discount. Each lever solves that for
    zero, holding everything else fixed. Negative changes mean there is a cushion.
    """
    paid_visits = max(net, 0) * margin
    return {
        "profit": net * margin - redemptions * discount,
        "discount": paid_visits / redemptions if redemptions > 0 else np.nan,
        # Fewer subsidized redemptions means fewer redemptions, incremental visits held fixed
        "subsidized_change": paid_visits / discount - redemptions,
        "subsidized": subsidized,
        "visit_change": redemptions * discount / margin - net,
        "net": net,
    }


# Within this share of discount spend, the promo counts as already at break even
BREAK_EVEN_BAND = 0.05
AT_BREAK_EVEN = "This promo is already at about break-even."


def about_break_even(levers: dict, redemptions: float, discount: float) -> bool:
    """True when profit is within BREAK_EVEN_BAND of the discount spend, or the break-even
    discount rounds to the current one, so the levers would show trivial changes."""
    near = abs(levers["profit"]) < BREAK_EVEN_BAND * redemptions * discount
    return near or round(levers["discount"], 2) == round(discount, 2)


def no_visits_break_even(runs: pd.DataFrame, discount: float, margin: float) -> str | None:
    """The break-even sentence when the planted promo adds no net visits, else None.

    Then no discount and no cut in subsidized redemptions can pay off, so the three
    levers would contradict "any one of these changes alone". It reads the true
    effect, like the levers, and counts net visits that round to 0 as none.
    """
    redemptions = runs["redemptions"].mean()
    if round(runs["true_net"].mean(), -1) > 0 or not redemptions > 0:
        return None
    needed = break_even_levers(0, redemptions, 0, discount, margin)["visit_change"]
    return ("With no extra visits, every redemption is a subsidy, so no discount level "
            f"breaks even. The promo would need about {nearest_ten(needed)} net incremental "
            "visits to pay for itself.")


def show_break_even(runs: pd.DataFrame, discount: float, margin: float):
    """Three levers that would each alone take the average true profit to zero.

    Near break even the levers would read like "cut $1.75 to $1.74", so one line
    replaces them. Any percentage is left out when its base is zero, and counts are
    rounded before their sign is chosen, so the text never divides by zero or shows "-0".
    """
    if (text := no_visits_break_even(runs, discount, margin)) is not None:
        st.markdown("#### What would make this promo break even?")
        st.markdown(no_math(text))
        return
    levers = break_even_levers(runs["true_net"].mean(), runs["redemptions"].mean(),
                               runs["true_subsidized"].mean(), discount, margin)
    profit, net, subsidized = levers["profit"], levers["net"], levers["subsidized"]
    change, more = levers["subsidized_change"], levers["visit_change"]
    st.markdown("#### What would make this promo break even?")
    if about_break_even(levers, runs["redemptions"].mean(), discount):
        st.markdown(no_math(f"With the true (planted) effects, the promo "
                            f"{'loses' if profit < 0 else 'makes'} about "
                            f"{money_text(abs(profit))} on average. {AT_BREAK_EVEN}"))
        return
    if profit < 0:
        text = (f"With the true (planted) effects, the promo loses about {money_text(-profit)} on "
                "average. Any one of these changes alone would bring it to break even.")
    else:
        text = (f"With the true (planted) effects, the promo makes about {money_text(profit)} on "
                "average. Here is how much room it has before losing money.")
    st.markdown(no_math(text))
    tiles = st.columns(3)

    tiles[0].metric("Break-even discount", dollars_text(levers["discount"]))
    if np.isnan(levers["discount"]):
        tiles[0].caption("No redemptions, so the discount costs nothing.")
    elif net <= 0:
        tiles[0].caption("No discount pays off: the promo adds no net visits.")
    elif profit < 0:
        tiles[0].caption(no_math(f"Cut the discount from {dollars_text(discount)} to "
                                 f"{dollars_text(levers['discount'])}."))
    else:
        tiles[0].caption(no_math(f"The discount could rise from {dollars_text(discount)} to "
                                 f"{dollars_text(levers['discount'])} before losing money."))

    percent = share_of(change, subsidized)
    if profit < 0:
        if subsidized > 0 and -change <= subsidized:
            tiles[1].metric("Fewer subsidized redemptions", percent)
            tiles[1].caption(f"Target offers away from regulars: {percent} fewer subsidized "
                             f"redemptions (about {nearest_ten(-change)} fewer).")
        else:
            tiles[1].metric("Fewer subsidized redemptions", "Not enough")
            tiles[1].caption("Even with no subsidized redemptions, the incremental visits "
                             "wouldn't cover the discounts.")
    else:
        # A percentage of zero subsidized redemptions is undefined, so show only the count
        tiles[1].metric("Room for more subsidized",
                        f"+{percent}" if percent else signed_ten(change))
        tiles[1].caption(f"Subsidized redemptions could rise "
                         + (f"{percent} (about {nearest_ten(change)} more)" if percent
                            else f"by about {nearest_ten(change)}")
                         + " before losing money.")

    percent = share_of(more, net)
    if profit < 0:
        tiles[2].metric("Extra visits needed", signed_ten(more))
        tiles[2].caption(f"Or the promo needs {nearest_ten(more)} more net incremental visits"
                         + (f" ({percent} more)." if percent else "."))
    else:
        tiles[2].metric("Room for fewer visits", signed_ten(more))
        tiles[2].caption(f"Net incremental visits could fall by {nearest_ten(-more)}"
                         + (f" (-{percent})" if percent and round(more, -1) else "")
                         + " before losing money.")


PLANTED_NOTE = ("The simulation plants a known true effect, then estimates it from noisy data "
                "the way a real test would. Estimated numbers are those noisy attempts; true "
                "numbers come from the planted effect, so the two can differ.")


def show_no_visits_headline(runs: pd.DataFrame, net_low: float, net_high: float):
    """The headline when the estimated net visits average zero or below.

    There is no cost per visit to state, and "About -56 net incremental visits, at
    n/a each" read as a broken page. Counts are rounded before formatting so they
    never read "-0".
    """
    def count(value):
        return f"{round(value) + 0.0:,.0f}"

    show_headline(
        "This promo added no measurable visits",
        f"Across {len(runs)} simulated promos, net incremental visits averaged about "
        f"{count(runs['did_net'].mean())} (most runs {count(net_low)} to {count(net_high)}), "
        f"so the {count(runs['redemptions'].mean())} redemptions mostly went to guests who "
        f"would have come anyway, at {dollars_text(runs['cost_per_redemption'].mean())} each.",
    )


def show_simulation(worlds, discount: float, margin: float):
    runs = worlds.runs
    net_low, net_high = middle_95(runs["did_net"])
    cost_per_visit = average(runs, "cost_per_visit")
    if np.isnan(cost_per_visit):
        show_no_visits_headline(runs, net_low, net_high)
    else:
        # Same net visits as the cost below, so cost / visits matches the stated cost per visit
        show_headline(
        f"About {runs['did_net'].mean():,.0f} net incremental visits, at "
        f"{dollars_text(cost_per_visit)} each",
        f"Across {len(runs)} simulated promos, {runs['redemptions'].mean():,.0f} redemptions "
        f"added about {runs['did_net'].mean():,.0f} net incremental visits on average "
        f"(most runs {net_low:,.0f} to {net_high:,.0f}). Each net incremental visit cost "
        f"about {dollars_text(cost_per_visit)}, vs. {dollars_text(runs['cost_per_redemption'].mean())} "
        "per redemption."
    )
    tiles = st.columns(4)
    tiles[0].metric("Average redemptions", f"{runs['redemptions'].mean():,.0f}")
    tiles[1].metric(
        "Average subsidized",
        f"{runs['subsidized'].mean():,.0f}",
        help="Redemptions that would have happened anyway during the promo window: "
        "redemptions minus the lift during the promo. Same definition as the "
        "Subsidized redemptions row below.",
    )
    tiles[2].metric(
        "Average estimated net profit",
        money_text(runs["net_profit"].mean()),
        help="The average of the estimates. Each run estimates profit from noisy data, "
        "as a real test would, so it can miss the true (planted) profit.",
    )
    tiles[3].metric(
        "Chance of true profit",
        f"{(runs['true_profit'] > 0).mean():.0%}",
        help="Share of simulated promos whose true profit was above zero.",
    )
    st.caption(PLANTED_NOTE)
    show_break_even(runs, discount, margin)

    st.markdown("#### Did it find the planted truth?")
    caption = "Tick a row to see its probability distribution."
    if not_meaningful(runs):
        caption += (f" {not_meaningful(runs)} of {len(runs)} runs estimated no net incremental "
                    "visits, so their cost per visit isn't meaningful and is left out of its range.")
    st.caption(caption)
    picked = st.dataframe(
        metrics_table(runs),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=view_key("sim_metrics"),
    )
    row = picked.selection.rows[0] if picked.selection.rows else 0
    try:
        show_chart(metric_distribution_chart(runs, row), "sim_metric")
    except ValueError:
        if METRICS[row][2] == "cost_per_visit" and runs["cost_per_visit"].isna().all():
            st.info("No run had net incremental visits, so there's no cost per visit to draw.")
        else:
            st.info("These runs don't vary enough to draw a distribution.")
    caption = "Each simulated promo is one draw of luck. The curve shows where results land."
    undefined = runs[METRICS[row][2]].isna().sum()
    if undefined and METRICS[row][2] == "cost_per_visit":
        caption += (f" {undefined} runs had no net incremental visits, so their cost per "
                    "visit is undefined and they aren't in the curve.")
    elif undefined:
        caption += f" {undefined} runs had no value and aren't in the curve."
    if METRICS[row][2] == "cost_per_visit" and np.isnan(average(runs, "true_cost_per_visit")):
        caption += (" The planted promo adds no net visits, so its true cost per visit is "
                    "undefined and has no line.")
    st.caption(caption)

    show_chart(weekly_means_chart(worlds.weekly_means, worlds.promo), "sim_weekly")
    title = f"Promo effect by week: average of {len(runs)} promos, bars show the middle 95%"
    clicked = show_chart(
        event_study_chart(worlds.study, worlds.promo, title),
        "sim_event_study",
        on_select="rerun",
        selection_mode="points",
    )
    show_clicked_week(clicked, simulated_week(worlds.weekly_gaps), "sim")

    with st.expander("What a single real test would report"):
        st.caption("A real promo only happens once, so this is what a marketer would see.")
        show_analysis(worlds.first_analysis, "one_world")


# Results for one promo


TERM_LABELS = {
    "net": "Net incremental visits",
    "promo_only": "Lift during the promo",
    "post_only": "After the promo (halo minus pull-forward)",
}


def no_math(text: str) -> str:
    """Escape dollar signs. Streamlit reads text between two $ signs as a math formula,
    which turned "$9.08 (range $4.67" into code-styled text with the signs gone."""
    return text.replace("$", "\\$")


def show_analysis(result, key: str, used: list | None = None):
    """Results for one promo. `used` lists the assumptions to show under the headline."""
    st.markdown(no_math(f"**{result.headline}**"))
    if used:
        with st.expander("Assumptions used"):
            st.markdown(no_math("\n".join(f"- {line}" for line in used)))
    if result.warnings:
        st.warning(no_math("**Read with care**\n" + "\n".join(f"- {w}" for w in result.warnings)))

    estimate = result.economics.set_index("metric")["estimate"]
    cost = estimate["cost_per_incremental_visit"]
    tiles = st.columns(4)
    tiles[0].metric("Redemptions", f"{estimate['redemptions']:,.0f}")
    tiles[1].metric(
        "Subsidized (would have come anyway)",
        f"{estimate['subsidized_redemptions']:,.0f}",
        help="Redemptions minus the lift during the promo.",
    )
    tiles[2].metric("Cost per net incremental visit", visit_cost_text(cost))
    tiles[3].metric("Net profit", money_text(estimate["net_profit"]))
    st.caption(no_math(visit_accounting(estimate)))

    ranges = result.ranges
    st.markdown("#### Estimates")
    st.caption(" ".join(filter(None, ["Tick a row to see its probability distribution.", ranges.note])))
    bounds = [interval(row) for _, row in result.did.iterrows()]
    table = pd.DataFrame(
        {
            "Quantity": result.did["term"].map(TERM_LABELS),
            "Estimate": result.did["estimate"].map(visits_text),
            f"Low ({ranges.label})": [visits_text(low) for low, _ in bounds],
            f"High ({ranges.label})": [visits_text(high) for _, high in bounds],
        }
    )
    picked = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=view_key(f"{key}_terms"),
    )
    rows = picked.selection.rows
    show_term_distribution(
        result.did.iloc[rows[0] if rows else 0], estimate["break_even_visits"], ranges, key
    )

    show_chart(result.weekly_chart, f"{key}_weekly")
    clicked = show_chart(
        result.event_study_chart,
        f"{key}_event_study",
        on_select="rerun",
        selection_mode="points",
    )
    show_clicked_week(clicked, estimated_week(result.event_study.coefficients), key)

    tiles = st.columns(2)
    detectable = result.detectable
    tiles[0].metric(
        "Smallest detectable net effect",
        f"{detectable['min_detectable_visits']:,.0f} visits" if detectable else "n/a",
    )
    tiles[1].metric("Break-even", f"{estimate['break_even_visits']:,.0f} visits")


def visit_accounting(estimate: pd.Series) -> str:
    """Show how redemptions, subsidized, promo-period and net visits add up.

    Uses rounded numbers that sum exactly. The after-promo part is net minus the
    promo-period lift, which matches the table's after-promo row when every store
    has every week, and comes close otherwise.
    """
    redemptions = round(estimate["redemptions"])
    gross = round(estimate["incremental_visits_promo"])
    subsidized = round(estimate["subsidized_redemptions"])
    net = round(estimate["net_incremental_visits"])
    if 0 <= gross <= redemptions:
        split = (f"{redemptions:,} redemptions = {gross:,} incremental visits during the promo "
                 f"+ {subsidized:,} subsidized.")
    else:
        split = (f"The lift during the promo ({gross:,}) is outside 0 to {redemptions:,} "
                 f"redemptions, so subsidized is held at {subsidized:,}.")
    after = net - gross
    sign = "+" if after >= 0 else "−"
    if after > 0:
        why = "so net is above the promo-period lift: halo outweighed pull-forward."
    elif after < 0:
        why = "so net is below the promo-period lift: pull-forward outweighed halo."
    else:
        why = "so net equals the promo-period lift."
    return (f"{split} Net incremental visits ({net:,}) = {gross:,} during the promo "
            f"{sign} {abs(after):,} after it (halo minus pull-forward), {why}")


def show_term_distribution(term: pd.Series, break_even: float, ranges, key: str):
    """Draw a smooth curve from one estimate's range, at the coverage the range is labelled with."""
    low, high = interval(term)
    if not (np.isfinite([low, high]).all() and high > low):
        st.info("This estimate has no range to draw a distribution from.")
        return
    label = TERM_LABELS[term["term"]]
    markers = [(0.0, "Zero")]
    if term["term"] == "net":
        markers.append((break_even, "Break-even"))
    x, density = density_from_interval(term["estimate"], low, high, ranges.share)
    chart = density_chart(x, density, f"{label}: likely range", label, markers, ",.0f")
    show_chart(chart, f"{key}_term")
    caption = (f"Smooth approximation peaking at the estimate, with {ranges.share:.0%} of it "
               f"inside this data's range.")
    if term["term"] == "net":
        chance = chance_above(break_even, term["estimate"], low, high, ranges.share)
        caption += (f" Approximate chance the promo paid for itself (area right of "
                    f"break-even): {chance:.0%}.")
    st.caption(caption)


# Import data tab


UPLOAD_KEY = "csv_upload"


KEPT = f"kept_{UPLOAD_KEY}"


def uploader_key() -> str:
    """The uploader's widget key. A new key empties it, which Streamlit can't do directly."""
    return f"{UPLOAD_KEY}_{st.session_state.get('uploader_round', 0)}"


def keep_upload(key: str):
    """Remember an uploaded file, so it survives a visit to another view.

    Streamlit empties an uploader that isn't drawn. This runs only when the viewer
    changes the uploader, so removing a file forgets it too.
    """
    upload = st.session_state[key]
    st.session_state[KEPT] = (upload.name, upload.getvalue()) if upload else None
    st.session_state.ran = False


def kept_upload(label: str, types) -> tuple | None:
    """An uploader that falls back to the file uploaded before leaving the view, or
    the sample loaded by Sample data. Returns (name, bytes), or None."""
    key = uploader_key()
    upload = st.file_uploader(label, type=types, key=key, on_change=keep_upload, args=(key,))
    if upload:
        return upload.name, upload.getvalue()
    kept = st.session_state.get(KEPT)
    if kept:
        st.caption(f"Using {kept[0]}")
    return kept


AI_PROMPT = """\
I need to reformat my restaurant's order data into a CSV for a promo analysis tool. \
Help me convert my export into exactly the format below.

One row per order. Include every order, not only discounted ones. Use these column \
names exactly, in lowercase:

Required:
- location_id: the store's ID. The same value on every order from that store.
- got_promo: "yes" if the store ran the promo, "no" if it didn't. The same value on \
every row for a store. Stores that didn't run the promo are needed as the comparison.
- timestamp: the order's local date and time, written as YYYY-MM-DD HH:MM:SS.
- discount_amount: the discount on the order in dollars, as a plain positive number \
like 3.00. Use 0 when there was no discount.

Recommended:
- market: the market or region the store is in (for example a city or DMA). The same \
value on every row for a store. If the promo ran in whole markets, every store in a promo \
market should have got_promo = yes.

Optional:
- promo_code: the code used on the order, blank if none. Useful when other discounts ran \
at the same time as the promo.
- guest_id, order_total, channel: keep them if my export has them.

Rules:
- At least 3 stores, some with got_promo = yes and some with no.
- History: at least 8 full weeks before the promo started, and at least as many weeks as \
the promo lasted.
- Keep orders through the end of the follow-up window after the promo (6 weeks by default).
- No currency symbols, thousands separators or extra header rows. Save as UTF-8 CSV.

My file is too large to paste in full, so don't retype the data. Give me a Python \
(pandas) script, or Excel or SQL steps, that converts the whole file. Ask me which stores \
ran the promo and which market each store is in if you can't tell from the data. At the \
end, remind me to enter the promo's first and last day and my average profit per order \
on the upload page.

Here are the column names and first 10 rows of my export:
[paste them here, with names and other personal guest details removed]"""

SETTINGS = {
    "promo_start": None,
    "promo_end": None,
    "margin": None,
    "post_weeks": 6,
    "promo_codes": "",
}


def assumptions(result, discount_per_redemption: float) -> list:
    """What the analysis assumed, in plain words, from the settings and the data."""
    config = result.config
    promo = set(config.treated_location_ids)
    if config.location_market:
        promo_markets = {m for location, m in config.location_market.items() if location in promo}
        control_markets = set(config.location_market.values()) - promo_markets
        groups = (f"{len(promo_markets)} promo and {len(control_markets)} control markets "
                  f"({len(promo)} promo stores)")
    else:
        groups = f"{len(promo)} promo stores, with every other store as a control (no market map)"
    post_end = config.window_end - pd.Timedelta(days=1)
    return [
        f"Margin per order: {dollars_text(config.avg_margin_per_order)}",
        f"Discount per redemption: {dollars_text(discount_per_redemption)} on average, "
        "from the data",
        f"{groups}",
        f"Promo: {config.promo_start:%b %d, %Y} to {config.promo_end:%b %d, %Y} "
        f"({config.promo_days} days, measured as {config.promo_weeks} weeks)",
        f"Post window: {config.post_window_weeks} weeks after the promo, to {post_end:%b %d, %Y}",
    ]


def run_analysis():
    st.session_state.ran = True
    st.session_state.just_ran = True


# The analysis is cached, so a repeat run is instant. A short pause shows it ran.
MIN_RUN_SECONDS = 1.0


SAMPLE_NAME = "example_orders.csv"
DEMO_HINT = "← Click here for demo"
SAMPLE_LOADED_HINT = "Sample loaded. Now click **Run analysis** below ↓"
RUN_HINT = "← Click Run analysis to see the results"


def load_sample():
    """Fill the page with the example file and its promo settings, ready to run."""
    csv_bytes, settings = example_upload()
    st.session_state[KEPT] = (SAMPLE_NAME, csv_bytes)
    st.session_state.uploader_round = st.session_state.get("uploader_round", 0) + 1
    for key, value in settings.items():
        st.session_state[f"setting_{key}"] = value
    st.session_state.ran = False


def promo_settings() -> dict:
    """The promo details the CSV can't hold. A setting that can't work is flagged
    under its field as soon as it's entered."""
    first, last, margin, weeks = st.columns(4)
    entered = {
        "promo_start": first.date_input("Promo's first day", key="setting_promo_start",
                                        format="YYYY-MM-DD"),
        "promo_end": last.date_input("Promo's last day", key="setting_promo_end",
                                     format="YYYY-MM-DD"),
        "margin": margin.number_input(
            "Average profit per order ($)", min_value=0.0, step=0.5, format="%.2f",
            key="setting_margin",
            help="Profit on a typical order before any discount. Drives net profit and "
                 "break even."),
        "post_weeks": weeks.number_input(
            "Weeks to watch after the promo", min_value=1, max_value=26, step=1,
            key="setting_post_weeks",
            help="Longer catches more lasting effects and pull-forward, but adds noise."),
    }
    entered["promo_codes"] = st.text_input(
        "Promo codes (optional, comma separated)", key="setting_promo_codes",
        help="Only orders with these codes count as redemptions. Leave blank to count every "
             "discounted order at promo stores during the promo.")
    columns = {"promo_end": last, "margin": margin, "post_weeks": weeks}
    for key, problem in setting_problems(entered).items():
        columns[key].error(problem)
    return entered


# check_settings names fields by read_upload's arguments; the page by SETTINGS keys
SETTING_FIELDS = {"promo_end": "promo_end", "avg_margin_per_order": "margin",
                  "post_window_weeks": "post_weeks"}


def setting_problems(entered: dict) -> dict:
    """Settings that can't work, by SETTINGS key. Blank fields are left to
    missing_inputs."""
    found = check_settings(entered["promo_start"], entered["promo_end"],
                           entered["margin"], entered["post_weeks"])
    return {SETTING_FIELDS[field]: problem for field, problem in found.items()
            if entered[SETTING_FIELDS[field]] is not None}


def missing_inputs(has_file: bool, entered: dict) -> list:
    """What still has to be filled in before the upload can be read, in words."""
    blank = [label for key, label in [("promo_start", "promo's first day"),
                                      ("promo_end", "promo's last day"),
                                      ("margin", "profit per order")]
             if entered[key] is None]
    return ([] if has_file else ["a CSV"]) + [f"the {label}" for label in blank]


def upload_settings(entered: dict) -> tuple:
    """read_upload's promo settings, from the Import page's inputs."""
    codes = tuple(code.strip() for code in entered["promo_codes"].split(",") if code.strip())
    return (entered["promo_start"], entered["promo_end"], float(entered["margin"]),
            int(entered["post_weeks"]), codes)


SETTINGS_HEADING = "Check these settings:"
FILE_HEADING = "This file can't be analyzed yet. Fix the following and upload it again:"


def show_problems(error: DataError):
    """Why the upload can't be analyzed: settings to change apart from fixes to the file."""
    for heading, problems in [(SETTINGS_HEADING, error.setting_problems),
                              (FILE_HEADING, error.file_problems)]:
        if problems:
            st.error(no_math(heading + "\n" + "\n".join(f"- {problem}" for problem in problems)))


def show_import():
    # The sample is the obvious first step until the viewer has a file of their own
    has_file = bool(st.session_state.get(KEPT))
    ran = st.session_state.get("ran")
    sample_loaded = has_file and st.session_state[KEPT][0] == SAMPLE_NAME
    # Settings as last entered: the inputs are drawn further down
    flagged = bool(setting_problems({key: st.session_state.get(f"setting_{key}")
                                     for key in SETTINGS}))
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.button("Try sample data", type="secondary" if has_file else "primary",
                  on_click=load_sample, help="Fills in an example file and its promo settings.")
        # Walk a first-time visitor through the demo: load the sample, then run it
        if not has_file:
            st.markdown(f":blue[{DEMO_HINT}]")
        elif sample_loaded and not ran and not flagged:
            st.markdown(SAMPLE_LOADED_HINT)
    example_bytes, _ = example_upload()
    with st.expander("Format your CSV with AI"):
        st.caption("Paste this into ChatGPT or Claude, with a few rows of your export "
                   "(no guest names or contact details).")
        st.code(AI_PROMPT, language=None, wrap_lines=True)
        st.download_button("Download the example CSV", example_bytes, "example_orders.csv")

    upload = kept_upload("Orders (CSV)", "csv")
    entered = promo_settings()
    missing = missing_inputs(bool(upload), entered)
    listed = ", ".join(missing[:-1]) + " and " + missing[-1] if len(missing) > 1 else "".join(missing)
    flagged = bool(setting_problems(entered))
    blocked = bool(missing) or flagged
    # Primary only once it can run, so the sample button is the one highlighted before
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.button("Run analysis", type="secondary" if blocked else "primary",
                  disabled=blocked, on_click=run_analysis,
                  help=f"Add {listed} first." if missing
                  else "Fix the settings marked above first." if flagged else None)
        if not blocked and not ran:
            st.markdown(f":blue[{RUN_HINT}]")
    if blocked or not st.session_state.get("ran"):
        return
    csv_bytes = upload[1]

    # Right after a click, a status box under the button shows the run; later reruns
    # (any widget change) reuse the cached result without it
    just_ran = st.session_state.pop("just_ran", False)
    status = st.status("Running the analysis...") if just_ran else None
    try:
        started = time.monotonic()
        result = analyze_upload(csv_bytes, *upload_settings(entered))
        if status:
            time.sleep(max(0.0, MIN_RUN_SECONDS - (time.monotonic() - started)))
            status.update(label="Analysis done. Results are below.", state="complete")
    except DataError as error:
        if status:
            status.update(label="The analysis couldn't run", state="error")
        show_problems(error)
        return
    except Exception as error:
        if status:
            status.update(label="The analysis couldn't run", state="error")
        st.error(f"Couldn't read this file ({type(error).__name__}): {error}")
        return

    used = assumptions(
        result, result.economics.set_index("metric").loc["cost_per_redemption", "estimate"]
    )
    with st.sidebar:
        st.subheader("Assumptions")
        st.caption(no_math("\n".join(f"- {line}" for line in used)))
    st.header("Results", anchor=False)
    show_analysis(result, "your_data", used)


# Starbucks tab


# A wave with fewer not-yet-sent customers than this is too noisy to read
MIN_CONTROLS = 50

METHOD_LABELS = {"difference": "Simple difference", "did": "Difference-in-differences"}


def two_decimals(value) -> str:
    return "—" if pd.isna(value) else f"{value:,.2f}"


def three_decimals(value) -> str:
    return "—" if pd.isna(value) else f"{value:,.3f}"


def wave_table(waves: pd.DataFrame) -> pd.DataFrame:
    """The wave comparison per day, so waves with different windows can be compared.

    Each effect is a total over its wave's window, and the windows run from 3 to 7
    days, so totals and their ranges are divided by the window's length in days.
    """
    days = waves["window_hours"] / 24
    return pd.DataFrame(
        {
            "Offer wave (hour)": waves["wave_hour"].map("{:,.0f}".format),
            "Window (days)": days.map("{:,.1f}".format),
            "Recipients": waves["n_recipients"].map("{:,}".format),
            "Not-yet-sent controls": waves["n_not_yet"].map("{:,}".format),
            "Pre-period gap / day": (waves["pre_visit_gap"] / days).map(three_decimals),
            "Extra visits / customer / day": (waves["visit_effect"] / days).map(three_decimals),
            "95% CI / day": [f"{three_decimals(low)} to {three_decimals(high)}"
                             for low, high in zip(waves["ci_low"] / days, waves["ci_high"] / days)],
            "Extra spend / customer / day ($)": (waves["spend_effect"] / days).map(two_decimals),
            "Method": waves["method"].map(METHOD_LABELS),
        }
    )


def window_totals(waves: pd.DataFrame) -> pd.DataFrame:
    """The same effects as totals over each wave's whole window."""
    return pd.DataFrame(
        {
            "Offer wave (hour)": waves["wave_hour"].map("{:,.0f}".format),
            "Window (hrs)": waves["window_hours"].map("{:,.0f}".format),
            "Pre-period gap": waves["pre_visit_gap"].map(two_decimals),
            "Extra visits / customer": waves["visit_effect"].map(two_decimals),
            "95% CI": [f"{two_decimals(low)} to {two_decimals(high)}"
                       for low, high in zip(waves["ci_low"], waves["ci_high"])],
            "Extra spend / customer ($)": waves["spend_effect"].map(two_decimals),
        }
    )


def show_starbucks():
    st.caption(
        "Udacity simulated this dataset to mimic app behavior, so it shows the tool running "
        "on outside data. It is not evidence about real customers."
    )
    try:
        subsidy, waves = run_starbucks()
    except (FileNotFoundError, ValueError) as error:
        st.info(str(error))
        return

    tiles = st.columns(4)
    tiles[0].metric("Completed offers", f"{subsidy['completed_offers']:,}")
    tiles[1].metric("Never viewed", f"{subsidy['completed_never_viewed']:,}")
    tiles[2].metric("Viewed only after earning", f"{subsidy['completed_before_viewing']:,}")
    tiles[3].metric(
        "Pure-subsidy rewards",
        f"{subsidy['unviewed_share_of_reward_cost']:.1%}",
        help="Share of reward cost that was pure subsidy: rewards paid on completed offers "
        "the customer never viewed, or viewed only after earning the reward. The offer "
        "can't have changed what they bought.",
    )

    st.markdown("#### First-time offer recipients vs. customers not yet sent an offer")
    readable = waves["n_not_yet"] >= MIN_CONTROLS
    # A static table wraps to the page width, where the grid would cut off columns
    st.table(wave_table(waves[readable]), hide_index=True)
    st.caption(
        "The waves' windows differ in length (each runs until the next wave, 3 to 7 days), "
        "so the effects are shown per customer per day, which can be compared across "
        "waves. Extra visits / customer / day is the recipients' extra visits each day, "
        "compared with customers not yet sent an offer. Wave 0 uses a simple difference "
        "because nothing comes before it to measure a pre-period; later waves use "
        "difference-in-differences, subtracting each group's visits in the same-length "
        "window before."
    )
    with st.expander("Totals over each window"):
        st.table(window_totals(waves[readable]), hide_index=True)
    if not readable.all():
        st.markdown(f"**Too small to read** (fewer than {MIN_CONTROLS} not-yet-sent controls)")
        small = wave_table(waves[~readable])
        st.table(small.style.set_properties(color=FAINT_TEXT), hide_index=True)


# Page


# The site opens on the first view
VIEWS = ["Import data", "Simulated promo", "Starbucks offers"]

# Key, label, lowest, highest (None for no limit), default, step, and what to divide the
# entry by to get the value planted_effects takes. In the order planted_effects takes them.
# Shares are typed as percentages, so 10 means 10%. Whole-number defaults make
# whole-number inputs. The $1.75 discount (REFERENCE_DISCOUNT) puts the default promo
# just under break even, with about a 40% chance of true profit, by request. The
# break-even card then shows its one-line "already at about break-even" message.
INPUTS = [
    ("discount", "Discount per redemption ($)", 0.01, None, REFERENCE_DISCOUNT, 0.25, 1),
    ("margin", "Margin per order ($)", 0.01, None, 6.0, 0.5, 1),
    ("occasional_lift", "Visit lift: occasional guests (%)", 0.0, 1000.0, 10.0, 5.0, 100),
    ("lapsed_lift", "Visit lift: lapsed guests (%)", 0.0, 1000.0, 100.0, 25.0, 100),
    ("new_lift", "Visit lift: new guests (%)", 0.0, 1000.0, 25.0, 5.0, 100),
    ("elasticity", "Discount elasticity", 0.0, 5.0, 0.0, 0.25, 1),
    ("subsidized_share", "Baseline visits that redeem anyway (%)", 0.0, 100.0, 20.0, 5.0, 100),
    ("n_guests", "Guests in the chain", 1_000, MAX_GUESTS, SynthConfig().n_guests, 5_000, 1),
    ("start_week", "Promo starts (weeks into the year)", 8, 45, 24, 1, 1),
    ("promo_weeks", "Promo length (weeks)", 1, 12, 4, 1, 1),
]
PERCENT_HELP = {
    "occasional_lift": "Extra visits during the promo, as a percentage of normal: 10 means 10% more.",
    "lapsed_lift": "Extra visits during the promo, as a percentage of normal: 100 means twice as many.",
    "new_lift": "Extra visits during the promo from guests in their first months: 25 means 25% "
                "more. Regulars have no lift.",
    "elasticity": "How much a bigger discount raises the lifts. Each lift is multiplied by "
                  "(discount ÷ reference discount) ^ elasticity. The reference is \\$1.75, or "
                  "your past promo's discount with Use my data. 0 means the discount size "
                  "doesn't change visits; 1 means doubling the discount doubles the lifts.",
    "subsidized_share": "Percentage of ordinary promo-period visits at promo stores that use the discount.",
    "n_guests": "Everyone who visits at least once in the year, spread over the stores. More "
                "guests means more redemptions and visits, and steadier estimates. With Use my "
                f"data it starts at your file's count; the simulation runs at most {MAX_GUESTS:,}.",
    "start_week": "When the discount begins. The simulated year is busiest in late summer.",
    "promo_weeks": "How many weeks the discount runs.",
}


USE_MY_DATA = "use_my_data"


def my_data_calibration():
    """The calibration for the Import page's file and settings, or None while they're
    incomplete. Raises DataError if the file can't be read."""
    upload = st.session_state.get(KEPT)
    entered = {key: st.session_state.get(f"setting_{key}") for key in SETTINGS}
    if missing_inputs(bool(upload), entered):
        return None
    return calibrate_upload(upload[1], *upload_settings(entered))


def fill_from_my_data():
    """When Use my data turns on, start the sliders at what the file says, once.

    Later reruns leave them alone, so the viewer can edit them.
    """
    if not st.session_state.get(USE_MY_DATA):
        return
    try:
        learned = my_data_calibration()
    except (DataError, ValueError):
        st.session_state[USE_MY_DATA] = False
        st.session_state.my_data_problem = True
        return
    if learned is None:
        return
    bounds = {key: (low, high) for key, _, low, high, *_ in INPUTS}
    for key, value in {"discount": learned.discount, "margin": learned.margin,
                       "n_guests": learned.overrides.get("n_guests", SynthConfig().n_guests),
                       **(learned.lifts or {})}.items():
        low, high = bounds[key]
        value = min(max(float(value), low), high or float("inf"))
        st.session_state[key] = int(round(value)) if key == "n_guests" else round(value, 2)


def my_data_toggle():
    """The Use my data switch and what was learned. Returns the calibration in use, or None."""
    ready = not missing_inputs(bool(st.session_state.get(KEPT)),
                               {key: st.session_state.get(f"setting_{key}") for key in SETTINGS})
    learned = None
    if not ready:
        st.session_state[USE_MY_DATA] = False
    elif st.session_state.get(USE_MY_DATA):
        try:
            learned = my_data_calibration()
        except (DataError, ValueError):
            st.session_state[USE_MY_DATA] = False
            st.session_state.my_data_problem = True
    st.toggle("Use my data", key=USE_MY_DATA, disabled=not ready, on_change=fill_from_my_data,
              help="Simulate your own stores, visits and guests, learned from your file."
              if ready else "Upload a CSV and enter the promo details on Import data first.")
    if st.session_state.pop("my_data_problem", False):
        st.warning("Your file can't be read. Check it on Import data.")
    if learned:
        st.caption(no_math("\n".join(f"- {note}" for note in learned.notes)))
    return learned


def limits_text(low, high, default) -> str:
    """The allowed range of an input, in words."""
    show = (lambda value: f"{value:,}") if isinstance(default, int) else (lambda value: f"{value:g}")
    return f"between {show(low)} and {show(high)}" if high is not None else f"at least {show(low)}"


def checked_inputs() -> tuple:
    """Draw the sidebar inputs and return (values, problems).

    The inputs have no min or max of their own: Streamlit rejects an out-of-range
    entry in the browser without telling the script, so the page kept showing old
    results with no explanation. Each entry is checked here instead. An entry
    outside its limits is reported, and the last valid value is used until it's fixed.
    """
    values, problems = [], []
    for key, label, low, high, default, step, scale in INPUTS:
        typed = st.number_input(label, step=step, key=key, help=PERCENT_HELP.get(key),
                                format="%d" if isinstance(default, int)
                                else "%.2f" if scale == 1 else "%.1f")
        valid = typed is not None and typed >= low and (high is None or typed <= high)
        if valid:
            st.session_state[f"valid_{key}"] = typed
        else:
            kept = st.session_state.get(f"valid_{key}", default)
            problems.append(f"{label} must be {limits_text(low, high, default)}. "
                            f"Using {kept:g} until it's fixed.")
        values.append((typed if valid else st.session_state.get(f"valid_{key}", default)) / scale)
    for problem in problems:
        st.warning(no_math(problem))
    return values, problems


def promo_timeline(start_week: int, promo_weeks: int) -> tuple:
    """Fit the promo into the simulated year. Returns (start week, note on any move).

    The analysis needs a baseline before the promo at least as long as the promo
    (and at least MIN_PRE_WEEKS), and the post window has to end within the year.
    """
    config = SynthConfig()
    earliest = max(MIN_PRE_WEEKS, promo_weeks)
    latest = config.n_weeks - config.post_window_weeks - promo_weeks
    fitted = min(max(start_week, earliest), latest)
    if fitted > start_week:
        return fitted, (f"Moved to week {fitted}: a {promo_weeks}-week promo needs at least "
                        f"{earliest} weeks of history before it.")
    if fitted < start_week:
        return fitted, (f"Moved to week {fitted}: the {promo_weeks}-week promo and its "
                        f"{config.post_window_weeks}-week follow-up must end within the "
                        f"{config.n_weeks}-week year.")
    return fitted, ""


def timeline_text(start_week: int, promo_weeks: int) -> str:
    """The promo and post window as dates in the simulated year."""
    config = SynthConfig()
    first = pd.Timestamp(config.start_date)
    start = first + pd.Timedelta(weeks=start_week)
    end = start + pd.Timedelta(weeks=promo_weeks) - pd.Timedelta(days=1)
    measured = end + pd.Timedelta(weeks=config.post_window_weeks)
    return (f"Discount runs {start:%b %d} to {end:%b %d, %Y}; visits are measured "
            f"through {measured:%b %d}.")


def main():
    st.set_page_config(page_title="Bikky demo: promo incrementality", layout="wide")
    st.html(css())
    st.markdown("**Bikky demo**")
    st.title("Promo incrementality")
    st.markdown(
        "Of a promo's redemptions, how many visits did it really add, and what did each one cost?"
    )

    if "first_seed" not in st.session_state:
        st.session_state.first_seed = 0
        st.session_state.view = 0
    # Streamlit forgets a widget's value while it isn't drawn. Writing each input's
    # value back keeps it while another view is showing.
    for key, *_, default, _, _ in INPUTS:
        st.session_state[key] = st.session_state.get(key, default)
    for key, default in SETTINGS.items():
        st.session_state[f"setting_{key}"] = st.session_state.get(f"setting_{key}", default)
    st.session_state[USE_MY_DATA] = st.session_state.get(USE_MY_DATA, False)

    # A switcher rather than st.tabs, so the sidebar knows which view is showing and
    # only offers the simulation controls where they do something
    shown = st.segmented_control("View", VIEWS, default=VIEWS[0], key="shown",
                                 label_visibility="collapsed") or VIEWS[0]
    simulated = shown == "Simulated promo"

    with st.sidebar:
        if simulated:
            # Filled in once the inputs below are read, so it states this run's guest count
            intro = st.empty()
            if st.button("Resimulate", type="primary", width="stretch"):
                st.session_state.first_seed = secrets.randbelow(1_000_000)
        if st.button("Reset graphs", type="tertiary", help="Clear zoom and selections."):
            st.session_state.view += 1
        if shown == "Import data":
            st.caption(
                "1. Upload a CSV of orders.\n"
                "2. Enter the promo details.\n"
                "3. Run analysis."
            )
        if simulated:
            learned = my_data_toggle()
            entered, problems = checked_inputs()
            *effects, start_week, promo_weeks = entered
            intro.caption(f"Plant the true effects, then see whether the estimates follow. "
                          f"Results average {N_WORLDS} simulated promos of a "
                          f"{int(effects[-1]):,}-guest chain.")
            start_week, moved = promo_timeline(int(start_week), int(promo_weeks))
            st.caption(" ".join(filter(None, [moved, timeline_text(start_week, int(promo_weeks))])))
            # Elasticity scales the lifts relative to the discount they were set for
            reference = learned.discount if learned else REFERENCE_DISCOUNT
            settings = (*effects, start_week, int(promo_weeks), reference)

    if simulated:
        if problems:
            st.warning("Some sidebar entries are out of range, so these results use the last "
                       "valid values. See the sidebar.")
        if learned:
            st.caption(f"Simulating your business: learned from {st.session_state[KEPT][0]}.")
        worlds = simulate(st.session_state.first_seed,
                          as_json(learned.overrides) if learned else "{}", *settings)
        show_simulation(worlds, *effects[:2])
    elif shown == "Import data":
        show_import()
    else:
        show_starbucks()


# The guard stops simulation worker processes from re-running the page when they
# import this file.
if __name__ == "__main__":
    main()
