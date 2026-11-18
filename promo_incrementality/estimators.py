"""Promo effect estimators. All of them read the common schema plus a PromoConfig.

Units: every effect is a count of visits across all treated locations, unless a
column says otherwise. `pre_post`, `did` and `economics` return tidy DataFrames
with an estimate, a confidence interval and notes for each row.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from promo_incrementality.formatting import dollars_text, visits_text
from promo_incrementality.schema import PromoConfig, validate_transactions

CONFIDENCE = 0.95
MIN_CLUSTERS = 3  # below this, cluster-robust inference can't work at all
# How often the pre-trend drift test fired on simulated data with parallel trends
# and five markets. See event_study().
PRETREND_FALSE_ALARM_RATE = 0.12
WEBB_WEIGHTS = np.sqrt(np.array([0.5, 1.0, 1.5]))


def weekly_panel(transactions: pd.DataFrame, config: PromoConfig) -> pd.DataFrame:
    """Build a location-by-week panel of visits.

    `week` counts weeks from promo start, so week 0 is the first promo week. The
    panel ends with the post window: later weeks still carry promo effects, and
    keeping them would make them look like an untreated baseline. Each location
    starts at its first full week of data, so a store that opened partway through
    the history isn't padded with weeks of zero visits from before it existed.
    After that, a week with no visits counts as zero.
    """
    validate_transactions(transactions)
    last_week = config.promo_weeks + config.post_window_weeks - 1
    if transactions["timestamp"].max().normalize() < config.last_day_needed:
        raise ValueError("transactions end before the post window does")
    days = (transactions["timestamp"].dt.normalize() - config.promo_start).dt.days
    week = days // 7
    first_full_week = -(-days.groupby(transactions["location_id"]).min() // 7)

    visits = (
        transactions.groupby(["location_id", week.rename("week")])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(first_full_week.min(), last_week + 1), fill_value=0)
        .stack()
        .rename("visits")
        .reset_index()
    )
    visits = visits[visits["week"] >= visits["location_id"].map(first_full_week)]
    visits["treated"] = visits["location_id"].isin(config.treated_location_ids).astype(int)
    visits["promo"] = visits["week"].between(0, config.promo_weeks - 1).astype(int)
    visits["post"] = (visits["week"] >= config.promo_weeks).astype(int)
    pre_promo = visits[visits["week"] < 0].groupby("location_id")["visits"].mean()
    visits["baseline"] = visits["location_id"].map(pre_promo)
    return visits


def _add_outcome(panel: pd.DataFrame, outcome: str) -> tuple:
    """Add the regression outcome `y`, and the visits one unit of `y` is worth.

    "indexed" divides each location's visits by its own pre-promo weekly mean.
    Seasonality and trend scale demand proportionally, so a big store moves more
    visits than a small one and raw counts can't have parallel trends unless
    treated and control stores happen to match in size. Indexed visits can. An
    index point at a location is worth that location's baseline in visits, so the
    returned value converts effects back to visits per week across treated stores.

    "visits" uses raw counts. It is the textbook specification, and it is only
    trustworthy when treated and control locations are similar in size.
    """
    treated = panel[panel["treated"] == 1].drop_duplicates("location_id")
    if outcome == "indexed" and not (panel["baseline"] > 0).all():
        raise ValueError("a location has no pre-promo visits, so it can't be indexed")
    if outcome == "indexed":
        return panel.assign(y=panel["visits"] / panel["baseline"]), treated["baseline"].sum()
    if outcome == "visits":
        return panel.assign(y=panel["visits"]), len(treated)
    raise ValueError(f"unknown outcome: {outcome}")


def clusters_by_market(mapping: dict | None, locations) -> bool:
    """Whether to cluster by market: only with a map giving at least MIN_CLUSTERS markets.

    The single rule behind did(), the power calculator and the app's warnings.
    """
    if not mapping:
        return False
    return len({mapping.get(location) for location in locations} - {None}) >= MIN_CLUSTERS


def _cluster_ids(panel: pd.DataFrame, config: PromoConfig, cluster_by: dict | None):
    """Integer cluster ids: `cluster_by`, else the config's markets, else locations."""
    mapping = cluster_by or config.location_market
    if mapping:
        missing = sorted(set(panel["location_id"]) - set(mapping))
        if missing:
            raise ValueError(f"no market given for locations: {missing}")
    if not clusters_by_market(mapping, panel["location_id"].unique()):
        return pd.factorize(panel["location_id"])[0]
    return pd.factorize(panel["location_id"].map(mapping))[0]


def promo_redemptions(transactions: pd.DataFrame, config: PromoConfig) -> pd.DataFrame:
    """Discounted orders at promo locations during the promo.

    Limited to `config.promo_codes` when set, so other discounts aren't counted.
    """
    days = transactions["timestamp"].dt.normalize()
    redeemed = (
        (transactions["discount_amount"] > 0)
        & transactions["location_id"].isin(config.treated_location_ids)
        & days.between(config.promo_start, config.promo_end)
    )
    if config.promo_codes:
        redeemed &= transactions["promo_code"].isin(config.promo_codes)
    return transactions[redeemed]


def pre_post(
    transactions: pd.DataFrame, config: PromoConfig, panel: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Treated-location visits in the promo vs. the same-length window just before.

    The CI uses the spread of location-level changes. It reflects sampling noise
    only, so it does nothing about the bias from seasonality and trend. `panel`
    reuses a `weekly_panel` built earlier.
    """
    panel = weekly_panel(transactions, config) if panel is None else panel
    treated = panel[panel["treated"] == 1]
    before_weeks = treated[treated["week"].between(-config.promo_weeks, -1)].groupby("location_id")
    history = before_weeks.size().reindex(treated["location_id"].unique(), fill_value=0)
    if (history < config.promo_weeks).any():
        raise ValueError("some promo locations have less history than the promo is long")
    during = treated[treated["promo"] == 1].groupby("location_id")["visits"].sum()
    before = before_weeks["visits"].sum()
    change = during - before
    n = len(change)
    estimate = change.sum()
    notes = "No control group, so seasonality and trend count as lift."
    if n < 2:
        margin = np.nan
        notes += " One promo location gives no range."
    else:
        margin = stats.t.ppf(0.5 + CONFIDENCE / 2, n - 1) * change.std(ddof=1) * np.sqrt(n)
    return _tidy(
        [
            {
                "estimator": "pre_post",
                "term": "promo_period",
                "estimate": estimate,
                "ci_low": estimate - margin,
                "ci_high": estimate + margin,
                "notes": notes,
            }
        ]
    )


def did(
    transactions: pd.DataFrame,
    config: PromoConfig,
    outcome: str = "indexed",
    cluster_by: dict | None = None,
    n_boot: int = 9999,
    seed: int = 0,
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Difference-in-differences on the weekly location panel.

    Fits y ~ treated:window + location FE + week FE. Standard errors are clustered
    by market when the config has a market map, because that is the level the promo
    was assigned at, and by location otherwise. `y` is indexed visits by default; see
    `_add_outcome` for why, and pass outcome="visits" for raw counts. Three rows
    come back:

        net         promo weeks plus post window, so pull-forward nets out
        promo_only  gross lift during the promo
        post_only   what happened afterward: halo minus pull-forward

    Estimates are total visits across treated locations over each window.

    `ci_low`/`ci_high` are the analytic clustered CI. `wild_ci_low`/`wild_ci_high`
    come from a wild cluster bootstrap, which is more reliable with few clusters.
    `cluster_by` maps location_id to a cluster and overrides the config. `n_boot=0`
    skips the bootstrap, leaving its columns empty. `panel` reuses a `weekly_panel`
    built earlier.
    """
    if config.post_window_weeks < 1:
        raise ValueError("post_window_weeks must be at least 1")
    panel = weekly_panel(transactions, config) if panel is None else panel
    panel, visits_per_unit = _add_outcome(panel, outcome)
    panel["window"] = panel["treated"] * (panel["promo"] + panel["post"])
    panel["promo_effect"] = panel["treated"] * panel["promo"]
    panel["post_effect"] = panel["treated"] * panel["post"]
    groups = _cluster_ids(panel, config, cluster_by)

    fixed_effects = "C(location_id) + C(week)"
    fit_kwargs = {"cov_type": "cluster", "cov_kwds": {"groups": groups}, "use_t": True}
    net_fit = smf.ols(f"y ~ window + {fixed_effects}", data=panel).fit(**fit_kwargs)
    # One model separates the promo weeks from the post window.
    split_fit = smf.ols(f"y ~ promo_effect + post_effect + {fixed_effects}", data=panel).fit(
        **fit_kwargs
    )
    specs = [
        ("net", net_fit, "window", config.promo_weeks + config.post_window_weeks,
         "Incremental minus pull-forward plus halo."),
        ("promo_only", split_fit, "promo_effect", config.promo_weeks,
         "Gross lift during the promo."),
        ("post_only", split_fit, "post_effect", config.post_window_weeks,
         "Halo minus pull-forward. Negative means pull-forward dominates."),
    ]

    rows = []
    for term, fit, regressor, n_weeks, notes in specs:
        scale = visits_per_unit * n_weeks
        low, high = fit.conf_int(alpha=1 - CONFIDENCE).loc[regressor]
        wild_low, wild_high = (
            wild_cluster_ci(fit, regressor, groups, n_boot, seed) if n_boot else (np.nan, np.nan)
        )
        rows.append(
            {
                "estimator": "did",
                "term": term,
                "estimate": fit.params[regressor] * scale,
                "ci_low": low * scale,
                "ci_high": high * scale,
                "wild_ci_low": wild_low * scale,
                "wild_ci_high": wild_high * scale,
                "notes": notes,
            }
        )
    return _tidy(rows)


@dataclass
class EventStudy:
    coefficients: pd.DataFrame  # week, estimate, ci_low, ci_high (per location-week)
    pretrend_p_value: float  # test that the pre-promo gap has no linear drift
    notes: str


def event_study(
    transactions: pd.DataFrame,
    config: PromoConfig,
    outcome: str = "indexed",
    n_pre_weeks: int = 12,
    panel: pd.DataFrame | None = None,
) -> EventStudy:
    """Treated-vs-control gap by week relative to the promo, with week -1 as the base.

    Coefficients are visits per treated location per week. Flat pre-promo
    coefficients support the parallel-trends assumption but cannot prove it.

    The pre-trend test asks whether the treated-vs-control gap drifted linearly
    before the promo. A joint F-test on every pre-promo coefficient is the usual
    choice, but with this few clusters it falsely rejected 40% of the time at the
    5% level on simulated data with parallel trends. This test rejected about 12% of
    the time with five markets (PRETREND_FALSE_ALARM_RATE), so a low p-value is a
    prompt to look, not a verdict.
    It only sees linear drift, so look at the chart too.
    """
    panel = weekly_panel(transactions, config) if panel is None else panel
    panel, visits_per_unit = _add_outcome(panel, outcome)
    panel = panel[panel["week"] >= -n_pre_weeks].copy()
    per_location = visits_per_unit / len(config.treated_location_ids)
    weeks = [w for w in sorted(panel["week"].unique()) if w != -1]
    names = {w: f"gap_{'m' if w < 0 else 'p'}{abs(w)}" for w in weeks}
    gaps = {
        name: (panel["treated"] * (panel["week"] == w)).astype(int)
        for w, name in names.items()
    }
    panel = panel.assign(**gaps)

    formula = f"y ~ {' + '.join(names.values())} + C(location_id) + C(week)"
    fit = smf.ols(formula, data=panel).fit(
        cov_type="cluster",
        cov_kwds={"groups": _cluster_ids(panel, config, None)},
        use_t=True,
    )
    intervals = fit.conf_int(alpha=1 - CONFIDENCE)
    coefficients = pd.DataFrame(
        {
            "week": weeks,
            "estimate": [fit.params[names[w]] * per_location for w in weeks],
            "ci_low": [intervals.loc[names[w], 0] * per_location for w in weeks],
            "ci_high": [intervals.loc[names[w], 1] * per_location for w in weeks],
        }
    )
    base = pd.DataFrame({"week": [-1], "estimate": [0.0], "ci_low": [0.0], "ci_high": [0.0]})
    coefficients = pd.concat([coefficients, base]).sort_values("week", ignore_index=True)

    pre_promo = panel[panel["week"] < 0].assign(
        drift=lambda p: p["treated"] * p["week"]
    )
    drift_fit = smf.ols("y ~ drift + C(location_id) + C(week)", data=pre_promo).fit(
        cov_type="cluster",
        cov_kwds={"groups": _cluster_ids(pre_promo, config, None)},
        use_t=True,
    )
    return EventStudy(
        coefficients,
        float(drift_fit.pvalues["drift"]),
        "Week -1 is the base, so its coefficient is zero by construction.",
    )


def economics(
    transactions: pd.DataFrame,
    config: PromoConfig,
    did_result: pd.DataFrame,
    redeemed: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Turn the DiD estimates into dollars, using the wild bootstrap CIs.

    When the bootstrap was skipped, the analytic CIs are used instead.

    Assumptions:
    - Every promo-period incremental visit redeemed, so subsidized redemptions are
      redemptions minus that gross lift.
    - `avg_margin_per_order` is before the discount. The discount is charged once,
      through total discount cost.
    - True cost divides total discount cost by net incremental visits. It is
      undefined wherever net incremental visits are zero or negative.
    """
    if redeemed is None:
        redeemed = promo_redemptions(transactions, config)
    redemptions = len(redeemed)
    discount_cost = redeemed["discount_amount"].sum()
    margin = config.avg_margin_per_order

    terms = did_result.set_index("term")
    net = np.array([terms.loc["net", "estimate"], *interval(terms.loc["net"])], dtype=float)
    gross = np.array(
        [terms.loc["promo_only", "estimate"], *interval(terms.loc["promo_only"])], dtype=float
    )

    def cost_per(visits):
        """Undefined for no net visits, and for an unbounded range end."""
        return discount_cost / visits if 0 < visits < np.inf else np.nan

    def finite(value):
        return value if np.isfinite(value) else np.nan

    def subsidized(gross_visits):
        return float(np.clip(redemptions - gross_visits, 0, redemptions))

    def per_redemption(cost):
        return cost / redemptions if redemptions else np.nan

    def metric(name, estimate, low=np.nan, high=np.nan, notes=""):
        return {"metric": name, "estimate": estimate, "ci_low": low, "ci_high": high, "notes": notes}

    return pd.DataFrame(
        [
            metric("redemptions", redemptions, notes="Observed."),
            metric("total_discount_cost", discount_cost, notes="Observed."),
            metric("incremental_visits_promo", *gross, "Gross, during the promo."),
            metric("subsidized_redemptions", subsidized(gross[0]),
                   subsidized(gross[2]), subsidized(gross[1]),
                   "Would have happened anyway. Kept between zero and all redemptions."),
            metric("net_incremental_visits", *net, "After pull-forward and halo."),
            metric("cost_per_redemption", per_redemption(discount_cost),
                   notes="Face value: what the promo appears to cost."),
            metric("cost_per_incremental_visit", cost_per(net[0]),
                   cost_per(net[2]), cost_per(net[1]),
                   "Total discount cost / net incremental visits."),
            metric("net_profit", net[0] * margin - discount_cost,
                   finite(net[1] * margin - discount_cost), finite(net[2] * margin - discount_cost),
                   "Net incremental visits x margin - total discount cost."),
            metric("break_even_visits", discount_cost / margin,
                   notes="Net incremental visits needed to pay back the discount."),
        ]
    )


def interval(row: pd.Series) -> tuple:
    """A did() row's wild bootstrap interval, or the analytic one if it was skipped."""
    if pd.notna(row.get("wild_ci_low", np.nan)):
        return row["wild_ci_low"], row["wild_ci_high"]
    return row["ci_low"], row["ci_high"]


def _range_text(low: float, high: float, text) -> str:
    """'a to b', 'at least a' or 'at most b', depending on which ends are known."""
    if np.isfinite(low) and np.isfinite(high):
        return f"{text(low)} to {text(high)}"
    if np.isfinite(low):
        return f"at least {text(low)}"
    if np.isfinite(high):
        return f"at most {text(high)}"
    return "unknown"


def headline(money: pd.DataFrame) -> str:
    """Write the result as one sentence, with ranges, from an `economics` table.

    The count and the cost both use net incremental visits, so total discount cost
    divided by the stated count gives the stated cost per visit. The cost leads with
    that point estimate, then its range.
    """
    rows = money.set_index("metric")
    cost = rows.loc["cost_per_incremental_visit"]
    net = rows.loc["net_incremental_visits"]

    visits, dollars = visits_text, dollars_text
    net_range = _range_text(net["ci_low"], net["ci_high"], visits)
    count = (f"Of {rows.loc['redemptions', 'estimate']:,.0f} redemptions, net incremental "
             f"visits came to about {net['estimate']:,.0f} ({net_range})")
    if net["ci_high"] < 0:
        cost_text = "so the promo lost visits overall"
    elif np.isnan(cost["estimate"]):
        cost_text = "so the data can't show it added any"
    else:
        cost_text = f"so each cost about {dollars(cost['estimate'])} ({_cost_range_text(cost)})"
    return f"{count}, {cost_text}, vs. {dollars(rows.loc['cost_per_redemption', 'estimate'])} per redemption."


def _cost_range_text(cost: pd.Series) -> str:
    """The cost per visit's range, saying why an end is missing.

    The upper end is missing when the visit range reaches zero, and the lower end
    when the visit range has no top.
    """
    low, high = cost["ci_low"], cost["ci_high"]
    if np.isnan(low) and np.isnan(high):
        return "too uncertain to give a range"
    if np.isnan(high):
        return f"range {dollars_text(low)} to no upper limit, since the visit range includes zero"
    if np.isnan(low):
        return f"at most {dollars_text(high)}"
    return f"range {dollars_text(low)} to {dollars_text(high)}"


def wild_cluster_ci(fit, term: str, groups, n_boot: int = 9999, seed: int = 0) -> tuple:
    """Wild cluster restricted bootstrap CI for one coefficient, by test inversion.

    A bootstrap test imposes a candidate value for the coefficient, flips the sign
    of each cluster's residuals at random, and asks how extreme the real t-statistic
    looks. The CI is every candidate value the test does not reject.

    Uses Rademacher weights, or Webb's six-point weights below 12 clusters, because
    Rademacher then has too few sign patterns to reach a 95% interval.
    """
    test = _WildClusterTest(fit, term, groups, n_boot, seed)
    alpha = 1 - CONFIDENCE
    center, half_width = test.estimate, test.std_error

    def boundary(direction):
        inside, outside = center, center + direction * half_width
        steps = 0
        while test.p_value(outside) >= alpha:
            inside, outside = outside, outside + direction * half_width
            steps += 1
            if steps > 100:  # very few clusters can't reject anything: unbounded
                return direction * np.inf
        for _ in range(30):
            middle = (inside + outside) / 2
            if test.p_value(middle) >= alpha:
                inside = middle
            else:
                outside = middle
        return inside

    return boundary(-1), boundary(1)


class _WildClusterTest:
    """Precomputes the pieces of the bootstrap so each candidate value is cheap.

    By Frisch-Waugh, one coefficient can be studied after partialling every other
    regressor out of y and x. The bootstrap t-statistics then only need each
    cluster's score, so no regression is refit inside the loop.
    """

    def __init__(self, fit, term, groups, n_boot, seed):
        design = pd.DataFrame(fit.model.exog, columns=fit.model.exog_names)
        others = design.drop(columns=term).to_numpy()
        self.basis = np.linalg.svd(others, full_matrices=False)[0][
            :, : np.linalg.matrix_rank(others)
        ]
        self.x = self._partial_out(design[term].to_numpy())
        self.y = self._partial_out(fit.model.endog)

        self.membership = (groups[:, None] == np.unique(groups)[None, :]).astype(float)
        self.x_by_cluster = self.membership * self.x[:, None]
        self.x_sq_by_cluster = self.x_by_cluster.T @ self.x
        self.xx = self.x @ self.x

        self.estimate = (self.x @ self.y) / self.xx

        # Residuals at a candidate c are y - c*x, so every per-cluster score is
        # linear in c. Partial out the regressors once here, not for each c.
        def cluster_scores(values):
            by_cluster = self.membership * values[:, None]
            return self.x_by_cluster.T @ self._partial_out(by_cluster), self.x_by_cluster.T @ values

        self.cross_y, self.own_y = cluster_scores(self.y)
        self.cross_x, self.own_x = cluster_scores(self.x)
        scores = self.x_by_cluster.T @ (self.y - self.estimate * self.x)
        self.score_norm = np.sqrt(scores @ scores)
        self.std_error = self.score_norm / self.xx

        n_clusters = self.membership.shape[1]
        rng = np.random.default_rng(seed)
        if n_clusters < 12:
            weights = np.concatenate([-WEBB_WEIGHTS, WEBB_WEIGHTS])
        else:
            weights = np.array([-1.0, 1.0])
        self.weights = rng.choice(weights, (n_boot, n_clusters))

    def _partial_out(self, values):
        return values - self.basis @ (self.basis.T @ values)

    def t_statistic(self, candidate):
        return (self.estimate - candidate) * self.xx / self.score_norm

    def p_value(self, candidate):
        """Two-sided bootstrap p-value for the null that the coefficient is `candidate`."""
        cross_scores = self.cross_y - candidate * self.cross_x
        own_scores = self.own_y - candidate * self.own_x

        delta = self.weights @ own_scores / self.xx
        boot_scores = self.weights @ cross_scores.T - delta[:, None] * self.x_sq_by_cluster
        boot_t = delta * self.xx / np.sqrt((boot_scores**2).sum(axis=1))
        return np.mean(np.abs(boot_t) >= abs(self.t_statistic(candidate)))


def _tidy(rows: list) -> pd.DataFrame:
    columns = ["estimator", "term", "estimate", "ci_low", "ci_high",
               "wild_ci_low", "wild_ci_high", "notes"]
    return pd.DataFrame(rows).reindex(columns=columns)
