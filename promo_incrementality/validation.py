"""Check the estimators against planted truth across many simulated worlds.

Run it directly to print the table:

    python -m promo_incrementality.validation
"""

import pandas as pd

from promo_incrementality.estimators import did
from promo_incrementality.montecarlo import map_in_workers, simulate_seeds
from promo_incrementality.synth import SynthConfig, generate


def run_recovery(
    seeds=range(20), outcome: str = "indexed", n_boot: int = 1999, **config_overrides
) -> pd.DataFrame:
    """Simulate once per seed and compare each estimator with its matching truth."""
    runs = simulate_seeds(seeds, n_boot=n_boot, outcome=outcome, **config_overrides).runs
    truth = runs["true_net"]
    return runs.assign(
        ci_covers=(runs["ci_low"] <= truth) & (truth <= runs["ci_high"]),
        wild_covers=(runs["wild_ci_low"] <= truth) & (truth <= runs["wild_ci_high"]),
    )


def summarize(recovery: pd.DataFrame) -> pd.Series:
    """Average performance across seeds. Errors are shares of the matching truth."""
    return pd.Series(
        {
            "mean_true_promo": recovery["true_promo"].mean(),
            "mean_pre_post": recovery["pre_post"].mean(),
            "pre_post_bias": recovery["pre_post"].mean() / recovery["true_promo"].mean() - 1,
            "mean_true_net": recovery["true_net"].mean(),
            "mean_did_net": recovery["did_net"].mean(),
            "did_net_bias": recovery["did_net"].mean() / recovery["true_net"].mean() - 1,
            "did_net_mean_abs_error": (
                (recovery["did_net"] - recovery["true_net"]).abs().mean()
                / recovery["true_net"].mean()
            ),
            "mean_true_post": recovery["true_post"].mean(),
            "mean_did_post": recovery["did_post"].mean(),
            "ci_coverage": recovery["ci_covers"].mean(),
            "wild_ci_coverage": recovery["wild_covers"].mean(),
            "median_wild_ci_width": (recovery["wild_ci_high"] - recovery["wild_ci_low"]).median(),
        }
    )


STRESS_SCENARIOS = [
    {"market_sizes": sizes, "market_shock_sd": sd, "market_shock_persistence": persistence}
    for sizes in [SynthConfig().market_sizes, (2,) * 10, (1,) * 20]
    for sd, persistence in [(0.0, 0.0), (0.02, 0.0), (0.02, 0.8)]
]


def _stress_one_seed(task: tuple) -> dict:
    """Check one simulated world against both clustering levels."""
    scenario, seed, n_boot = task
    result = generate(SynthConfig(seed=seed, **scenario))
    data, config = result.transactions, result.promo_config
    truth = result.ground_truth["net_incremental_visits"]
    row = {}
    by_location = {location: location for location in result.location_market}
    for name, clusters in [("location", by_location), ("market", result.location_market)]:
        net = did(data, config, cluster_by=clusters, n_boot=n_boot, seed=seed)
        net = net.set_index("term").loc["net"]
        row[f"{name}_covers"] = net["wild_ci_low"] <= truth <= net["wild_ci_high"]
        row[f"{name}_width"] = net["wild_ci_high"] - net["wild_ci_low"]
    row["error"] = net["estimate"] - truth
    return row


def clustering_stress_test(n_seeds: int = 500, n_boot: int = 999) -> pd.DataFrame:
    """Measure net-effect CI coverage across market counts and market-level shocks.

    The promo is assigned by market, so market shocks are what location clustering
    can miss. Every scenario keeps 20 locations. Uses all CPU cores and takes
    several minutes.
    """
    rows = []
    for scenario in STRESS_SCENARIOS:
        tasks = [(scenario, seed, n_boot) for seed in range(1000, 1000 + n_seeds)]
        runs = pd.DataFrame(map_in_workers(_stress_one_seed, tasks, chunksize=10))
        rows.append(
            {
                "markets": len(scenario["market_sizes"]),
                "shock_sd": scenario["market_shock_sd"],
                "persistence": scenario["market_shock_persistence"],
                "location_coverage": runs["location_covers"].mean(),
                "market_coverage": runs["market_covers"].mean(),
                "location_ci_width": runs["location_width"].median(),
                "market_ci_width": runs["market_width"].median(),
                "error_sd": runs["error"].std(),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    recovery = run_recovery()
    columns = ["seed", "true_promo", "pre_post", "true_net", "did_net",
               "wild_ci_low", "wild_ci_high", "wild_covers"]
    print(recovery[columns].round(0).to_string(index=False))
    print()
    print(summarize(recovery).round(3).to_string())
