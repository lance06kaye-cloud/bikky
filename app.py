"""Promo incrementality demo. Run with: streamlit run app.py"""

import html
import io
import secrets
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from promo_incrementality.analysis import analyze
from promo_incrementality.charts import (
    chance_above,
    density_chart,
    density_from_interval,
    density_from_samples,
    event_study_chart,
    weekly_means_chart,
)
from promo_incrementality.estimators import interval
from promo_incrementality.formatting import dollars_text, money_text, visits_text
from promo_incrementality.intake import (
    MIN_PRE_WEEKS,
    DataError,
    load_transactions,
    read_config,
    read_upload,
    to_upload,
)
from promo_incrementality.load_starbucks import load_starbucks, wave_comparison
from promo_incrementality.montecarlo import IN_BROWSER, run_worlds
from promo_incrementality.synth import SynthConfig

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


def planted_effects(
    discount, margin, occasional_lift, lapsed_lift, subsidized_share, start_week=24, promo_weeks=4
) -> dict:
    return {
        "promo_start_week": start_week,
        "promo_weeks": promo_weeks,
        "discount_amount": discount,
        "avg_margin_per_order": margin,
        "lift_by_segment": {
            **SynthConfig().lift_by_segment,
            "occasional": occasional_lift,
            "lapsed": lapsed_lift,
        },
        "subsidized_share": subsidized_share,
    }


# Cached as a shared resource, not copied on every rerun. The result is never changed.
@st.cache_resource(show_spinner=f"Simulating {N_WORLDS} promos...", max_entries=8)
def simulate(first_seed, *settings):
    return run_worlds(N_WORLDS, first_seed, analyze_first=True, **planted_effects(*settings))


@st.cache_resource(show_spinner=False, max_entries=4)
def analyze_upload(csv_bytes: bytes, *settings):
    """Analyze the one-file upload. `settings` are read_upload's promo settings."""
    return analyze(*read_upload(io.BytesIO(csv_bytes), *settings))


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
NAIVE_LABEL = "Naive before/after estimate of lift during the promo"
METRICS = [
    ("Net incremental visits", "true_net", "did_net", visits_text),
    ("Lift during the promo", "true_promo", "did_promo", visits_text),
    (NAIVE_LABEL, "true_promo", "pre_post", visits_text),
    ("After the promo (halo minus pull-forward)", "true_post", "did_post", visits_text),
    ("Subsidized redemptions", "true_subsidized", "subsidized", visits_text),
    ("Cost per net incremental visit", "true_cost_per_visit", "cost_per_visit", dollars_text),
    ("Net profit", "true_profit", "net_profit", money_text),
]
# The same rounding as each formatter above, for chart axes and hover (d3 formats)
CHART_FORMATS = {visits_text: ",.0f", dollars_text: "$,.2f", money_text: "$,.0f"}


def average(runs: pd.DataFrame, column: str) -> float:
    """Average a column across runs.

    Cost per visit is total cost over total visits. Averaging each run's ratio would
    let runs with almost no net visits dominate, and skip runs with none.
    """
    if column in ("cost_per_visit", "true_cost_per_visit"):
        visits = runs["did_net" if column == "cost_per_visit" else "true_net"].mean()
        return runs["discount_cost"].mean() / visits if visits > 0 else np.nan
    return runs[column].mean()


def metrics_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, truth, estimate, text in METRICS:
        low, high = middle_95(runs[estimate])
        rows.append(
            {
                "Average per promo": label,
                "Truth": text(average(runs, truth)),
                "Estimate": text(average(runs, estimate)),
                "Middle 95% of runs": f"{text(low)} to {text(high)}",
            }
        )
    return pd.DataFrame(rows)


def metric_distribution_chart(runs: pd.DataFrame, row: int):
    """Probability density of one METRICS row across the simulated promos.

    Raises ValueError when the runs don't vary enough to draw a curve.
    """
    label, truth, estimate, text = METRICS[row]
    markers = [(average(runs, truth), "True average")]
    if estimate == "did_net":
        markers.append((runs["break_even_visits"].mean(), "Break-even"))
    elif estimate == "net_profit":
        markers.append((0.0, "Break-even"))
    elif estimate == "cost_per_visit":
        markers.append((runs["cost_per_redemption"].mean(), "Face value"))
    x, density = density_from_samples(runs[estimate])
    title = f"{label}: distribution across {len(runs)} simulated promos"
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


def show_headline(text: str):
    """A headline styled like a heading, but a paragraph.

    Streamlit gives every heading an id built from its text, even with anchor=False,
    so a heading with numbers would leave stale numbers in a copied link.
    """
    st.html(
        '<p style="font-size: 1.5rem; font-weight: 600; line-height: 1.35; '
        f'margin: 0.25rem 0 1rem">{html.escape(text)}</p>'
    )


def show_simulation(worlds):
    runs = worlds.runs
    net_low, net_high = middle_95(runs["did_net"])
    cost_per_visit = average(runs, "cost_per_visit")
    # Same net visits as the cost below, so cost / visits matches the stated cost per visit
    show_headline(
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
    tiles[2].metric("Average net profit", money_text(runs["net_profit"].mean()))
    tiles[3].metric(
        "Chance of profit",
        f"{(runs['true_profit'] > 0).mean():.0%}",
        help="Share of simulated promos whose true profit was above zero.",
    )

    st.markdown("#### Did it find the planted truth?")
    st.caption(
        "Tick a row to see its probability distribution. The naive before/after row's "
        "truth is the actual lift during the promo, so the row shows how far a naive "
        "comparison with the weeks before misses it."
    )
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
        st.info("These runs don't vary enough to draw a distribution.")
    caption = "Each simulated promo is one draw of luck. The curve shows where results land."
    undefined = runs[METRICS[row][2]].isna().sum()
    if undefined and METRICS[row][2] == "cost_per_visit":
        caption += (f" {undefined} runs had no net incremental visits, so their cost per "
                    "visit is undefined and they aren't in the curve.")
    elif undefined:
        caption += f" {undefined} runs had no value and aren't in the curve."
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
    tiles[2].metric("Cost per net incremental visit", dollars_text(cost))
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


def load_sample():
    """Fill the page with the example file and its promo settings, ready to run."""
    csv_bytes, settings = example_upload()
    st.session_state[KEPT] = ("example_orders.csv", csv_bytes)
    st.session_state.uploader_round = st.session_state.get("uploader_round", 0) + 1
    for key, value in settings.items():
        st.session_state[f"setting_{key}"] = value
    st.session_state.ran = False


def promo_settings() -> dict:
    """The promo details the CSV can't hold."""
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
    return entered


def show_import():
    # The sample is the obvious first step until the viewer has a file of their own
    has_file = bool(st.session_state.get(KEPT))
    st.button("Try sample data", type="secondary" if has_file else "primary",
              on_click=load_sample, help="Fills in an example file and its promo settings.")
    example_bytes, _ = example_upload()
    with st.expander("Format your CSV with AI"):
        st.caption("Paste this into ChatGPT or Claude, with a few rows of your export "
                   "(no guest names or contact details).")
        st.code(AI_PROMPT, language=None, wrap_lines=True)
        st.download_button("Download the example CSV", example_bytes, "example_orders.csv")

    upload = kept_upload("Orders (CSV)", "csv")
    entered = promo_settings()
    blank = [label for key, label in [("promo_start", "promo's first day"),
                                      ("promo_end", "promo's last day"),
                                      ("margin", "profit per order")]
             if entered[key] is None]
    missing = ([] if upload else ["a CSV"]) + [f"the {label}" for label in blank]
    listed = ", ".join(missing[:-1]) + " and " + missing[-1] if len(missing) > 1 else "".join(missing)
    st.button("Run analysis", type="primary", disabled=bool(missing), on_click=run_analysis,
              help=f"Add {listed} first." if missing else None)
    if missing or not st.session_state.get("ran"):
        return
    csv_bytes = upload[1]

    codes = tuple(code.strip() for code in entered["promo_codes"].split(",") if code.strip())
    started = time.monotonic()
    try:
        with st.spinner("Running the analysis..."):
            result = analyze_upload(csv_bytes, entered["promo_start"], entered["promo_end"],
                                    float(entered["margin"]), int(entered["post_weeks"]), codes)
            if st.session_state.pop("just_ran", False) and not IN_BROWSER:
                time.sleep(max(0.0, MIN_RUN_SECONDS - (time.monotonic() - started)))
    except DataError as error:
        st.error("This file can't be analyzed yet. Fix the following and upload it again:\n"
                 + "\n".join(f"- {problem}" for problem in error.problems))
        return
    except Exception as error:
        st.error(f"Couldn't read this file ({type(error).__name__}): {error}")
        return

    used = assumptions(
        result, result.economics.set_index("metric").loc["cost_per_redemption", "estimate"]
    )
    with st.sidebar:
        st.subheader("Assumptions")
        st.caption(no_math("\n".join(f"- {line}" for line in used)))
    st.divider()
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
        st.divider()
        st.markdown(f"**Too small to read** (fewer than {MIN_CONTROLS} not-yet-sent controls)")
        small = wave_table(waves[~readable])
        st.table(small.style.set_properties(color="#9aa0a6"), hide_index=True)


# Page


# The site opens on the first view
VIEWS = ["Import data", "Simulated promo", "Starbucks offers"]

# Key, label, lowest, highest (None for no limit), default, step, and what to divide the
# entry by to get the value planted_effects takes. In the order planted_effects takes them.
# Shares are typed as percentages, so 10 means 10%. Whole-number defaults make
# whole-number inputs. The $1.75 discount makes the default promo about break even:
# average true profit near $0.
INPUTS = [
    ("discount", "Discount per redemption ($)", 0.01, None, 1.75, 0.25, 1),
    ("margin", "Margin per order ($)", 0.01, None, 6.0, 0.5, 1),
    ("occasional_lift", "Visit lift: occasional guests (%)", 0.0, 1000.0, 10.0, 5.0, 100),
    ("lapsed_lift", "Visit lift: lapsed guests (%)", 0.0, 1000.0, 100.0, 25.0, 100),
    ("subsidized_share", "Baseline visits that redeem anyway (%)", 0.0, 100.0, 20.0, 5.0, 100),
    ("start_week", "Promo starts (weeks into the year)", 8, 45, 24, 1, 1),
    ("promo_weeks", "Promo length (weeks)", 1, 12, 4, 1, 1),
]
PERCENT_HELP = {
    "occasional_lift": "Extra visits during the promo, as a percentage of normal: 10 means 10% more.",
    "lapsed_lift": "Extra visits during the promo, as a percentage of normal: 100 means twice as many.",
    "subsidized_share": "Percentage of ordinary promo-period visits at promo stores that use the discount.",
    "start_week": "When the discount begins. The simulated year has seasonality peaking in "
                  "late summer, so the start week changes how much a naive before/after "
                  "comparison is fooled.",
    "promo_weeks": "How many weeks the discount runs.",
}


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

    # A switcher rather than st.tabs, so the sidebar knows which view is showing and
    # only offers the simulation controls where they do something
    shown = st.segmented_control("View", VIEWS, default=VIEWS[0], key="shown",
                                 label_visibility="collapsed") or VIEWS[0]
    simulated = shown == "Simulated promo"

    with st.sidebar:
        if simulated:
            st.header("Simulated promo")
            st.caption(f"Plant the true effects, then see whether the estimates follow. Results "
                       f"average {N_WORLDS} simulated promos.")
            if st.button("Resimulate", type="primary", width="stretch"):
                st.session_state.first_seed = secrets.randbelow(1_000_000)
        if st.button("Reset graphs", width="stretch", help="Clear zoom and selections."):
            st.session_state.view += 1
        if shown == "Import data":
            st.header("Import data")
            st.caption(
                "1. Upload a CSV of orders.\n"
                "2. Enter the promo details.\n"
                "3. Run analysis."
            )
        if simulated:
            entered = [
                st.number_input(label, low, high, step=step, key=key, help=PERCENT_HELP.get(key),
                                format="%d" if isinstance(default, int)
                                else "%.2f" if scale == 1 else "%.1f") / scale
                for key, label, low, high, default, step, scale in INPUTS
            ]
            *effects, start_week, promo_weeks = entered
            start_week, moved = promo_timeline(int(start_week), int(promo_weeks))
            st.caption(" ".join(filter(None, [moved, timeline_text(start_week, int(promo_weeks))])))
            settings = (*effects, start_week, int(promo_weeks))

    if simulated:
        show_simulation(simulate(st.session_state.first_seed, *settings))
    elif shown == "Import data":
        show_import()
    else:
        show_starbucks()


# The guard stops simulation worker processes from re-running the page when they
# import this file.
if __name__ == "__main__":
    main()
