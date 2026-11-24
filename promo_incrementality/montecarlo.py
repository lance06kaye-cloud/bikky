"""Run the same simulated promo many times, to see the spread of outcomes.

One simulated world is one draw of luck. Averaging over many shows what the method
does in general, and the spread shows how far a single real test could land from
the truth.
"""

import atexit
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

import numpy as np
import pandas as pd

from promo_incrementality.analysis import Analysis, analyze
from promo_incrementality.charts import weekly_means
from promo_incrementality.estimators import (
    EventStudy,
    did,
    economics,
    event_study,
    pre_post,
    weekly_panel,
)
from promo_incrementality.schema import PromoConfig
from promo_incrementality.synth import SynthConfig, generate


@dataclass
class ManyWorlds:
    runs: pd.DataFrame  # one row per world: truth and estimates
    weekly_means: pd.DataFrame  # visits per location by week; columns 1=treated, 0=control
    study: EventStudy  # average gap by week; bounds are the middle 95% of worlds
    weekly_gaps: pd.DataFrame  # week and estimate, one row per world per week
    promo: PromoConfig  # from the first world; dates are the same in all of them
    first_analysis: Analysis | None  # full analysis of the first world, if requested


def _one_world(task: tuple) -> tuple:
    """Simulate one world and estimate it.

    `n_boot=0` skips the bootstrap to stay fast. With `full_analysis`, runs analyze()
    instead, bootstrap included, as one real test would report it, and reads the
    estimates from that.
    """
    seed, overrides, full_analysis, n_boot, outcome = task
    result = generate(SynthConfig(seed=seed, **overrides))
    data, config, truth = result.transactions, result.promo_config, result.ground_truth

    panel = weekly_panel(data, config)
    analysis = analyze(data, config) if full_analysis else None
    if analysis:
        results, money, before_after = analysis.did, analysis.economics, analysis.pre_post
        study = analysis.event_study.coefficients
    else:
        results = did(data, config, outcome=outcome, n_boot=n_boot, seed=seed, panel=panel)
        money = economics(data, config, results)
        before_after = pre_post(data, config, panel)
        study = event_study(data, config, outcome=outcome, panel=panel).coefficients

    effects = results.set_index("term")
    money = money.set_index("metric")["estimate"]
    cost = truth["total_discount_cost"]
    true_net = truth["net_incremental_visits"]
    row = {
        "seed": seed,
        "true_promo": truth["incremental_visits"],
        "true_post": truth["post_window_effect"],
        "true_subsidized": truth["subsidized_redemptions"],
        "true_net": true_net,
        "true_cost_per_visit": cost / true_net if true_net > 0 else np.nan,
        "true_profit": true_net * config.avg_margin_per_order - cost,
        "pre_post": before_after["estimate"].item(),
        "did_promo": effects.loc["promo_only", "estimate"],
        "did_post": effects.loc["post_only", "estimate"],
        "did_net": effects.loc["net", "estimate"],
        **{column: effects.loc["net", column]
           for column in ["ci_low", "ci_high", "wild_ci_low", "wild_ci_high"]},
        "redemptions": money["redemptions"],
        # Unclipped: economics() keeps one test's subsidized count between zero and all
        # redemptions, which near zero cuts off only the low misses and biases the
        # average upward. Averages across runs need the raw estimate.
        "subsidized": money["redemptions"] - effects.loc["promo_only", "estimate"],
        "discount_cost": money["total_discount_cost"],
        "cost_per_redemption": money["cost_per_redemption"],
        "cost_per_visit": money["cost_per_incremental_visit"],
        "net_profit": money["net_profit"],
        "break_even_visits": money["break_even_visits"],
    }
    return row, weekly_means(panel), study, config, analysis


def average_weekly_means(weeklies) -> pd.DataFrame:
    """Average each week over the worlds that have it, so a short world leaves no gap."""
    return pd.concat(weeklies).groupby(level=0).mean()


# In the browser (stlite on Pyodide) there are no extra processes to start
IN_BROWSER = sys.platform == "emscripten"

_pool = None
_pool_lock = threading.Lock()  # Streamlit runs each browser session on its own thread


def _workers() -> ProcessPoolExecutor:
    """One pool kept for the whole session, so repeat runs skip worker start-up."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ProcessPoolExecutor()
        return _pool


def _shut_down_workers(broken: ProcessPoolExecutor | None = None) -> None:
    """Shut the pool down. With `broken`, only if no other caller has replaced it yet."""
    global _pool
    with _pool_lock:
        if _pool is not None and (broken is None or _pool is broken):
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


atexit.register(_shut_down_workers)


def run_worlds(
    n_worlds: int = 100, first_seed: int = 0, analyze_first: bool = False, **config_overrides
) -> ManyWorlds:
    """Simulate `n_worlds` promos in parallel. `config_overrides` go to SynthConfig.

    `analyze_first` also runs the full analysis on the first world, in the same pass.
    """
    seeds = range(first_seed, first_seed + n_worlds)
    return simulate_seeds(seeds, analyze_first=analyze_first, **config_overrides)


def map_in_workers(function, tasks: list, chunksize: int) -> list:
    """Run `function` over `tasks` in the shared worker pool, or in turn in the browser."""
    if IN_BROWSER:
        return [function(task) for task in tasks]
    pool = _workers()
    try:
        return list(pool.map(function, tasks, chunksize=chunksize))
    except (BrokenProcessPool, RuntimeError):  # a worker died, or another session shut the pool
        _shut_down_workers(broken=pool)
        return list(_workers().map(function, tasks, chunksize=chunksize))


def simulate_seeds(
    seeds,
    analyze_first: bool = False,
    n_boot: int = 0,
    outcome: str = "indexed",
    **config_overrides,
) -> ManyWorlds:
    """Simulate one promo per seed in parallel, with the bootstrap if `n_boot` > 0."""
    seeds = list(seeds)
    tasks = [
        (seed, config_overrides, analyze_first and i == 0, n_boot, outcome)
        for i, seed in enumerate(seeds)
    ]
    results = map_in_workers(_one_world, tasks, chunksize=max(1, len(seeds) // 40))

    rows, weeklies, coefficients, configs, analyses = zip(*results)
    weekly_gaps = pd.concat(coefficients, ignore_index=True)[["week", "estimate"]]
    gaps = weekly_gaps.groupby("week")["estimate"]
    study = EventStudy(
        pd.DataFrame(
            {
                "estimate": gaps.mean(),
                "ci_low": gaps.quantile(0.025),
                "ci_high": gaps.quantile(0.975),
            }
        ).reset_index(),
        float("nan"),
        f"Average across {len(seeds)} worlds. Bars show the middle 95% of worlds.",
    )
    return ManyWorlds(
        runs=pd.DataFrame(rows),
        weekly_means=average_weekly_means(weeklies),
        study=study,
        weekly_gaps=weekly_gaps,
        promo=configs[0],
        first_analysis=analyses[0],
    )
