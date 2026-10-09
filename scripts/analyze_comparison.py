"""Regenerate the audited report and figures for the completed 20261008 OU study.

Run from an installed checkout:
    python scripts/analyze_comparison.py --input runs/comparison-full
No filters are rerun and the original run artifacts are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from likelihood_filtering.randomness import seeded_rng

LABELS = {
    "ou_linear_gaussian": "Linear Gaussian",
    "ou_nonlinear_gaussian": "Nonlinear Gaussian",
    "ou_poisson": "Poisson",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, label: str) -> None:
    require(bool(np.isclose(actual, expected, rtol=1e-12, atol=1e-12)), f"Mismatch: {label}")


def canonical(rows: list[dict[str, Any]]) -> Counter:
    return Counter(tuple(sorted((key, str(value)) for key, value in row.items())) for row in rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(root: Path) -> tuple[dict, list, list, list, dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    artifacts = []
    expected_rows, expected_costs = 0, 0
    for case in manifest["experiment"]["cases"]:
        for replicate in range(case["replicates"]):
            path = root / "records" / case["name"] / f"{replicate:04d}.json"
            artifact = json.loads(path.read_text())
            require(
                artifact["case"] == case["name"] and artifact["replicate"] == replicate,
                f"Record identity: {path}",
            )
            require(artifact["checkpoints"] == manifest["checkpoint_steps"], f"Checkpoints: {path}")
            base = "kalman" if case["name"] == "ou_linear_gaussian" else "particle"
            variants = [("score_root", "primary", 0), (base, "primary", 0)]
            if replicate < manifest["check_replicates"]:
                variants += [
                    ("score_root", "seed_repeat", i) for i in range(1, manifest["filter_repeats"])
                ]
                if base == "particle":
                    variants += [
                        (base, "seed_repeat", i) for i in range(1, manifest["filter_repeats"])
                    ]
                    variants += [(base, "particle_count", 0), (base, "grid_resolution", 0)]
            expected = {
                (method, variant, repeat, step)
                for method, variant, repeat in variants
                for step in manifest["checkpoint_steps"]
            }
            keys = [(r["method"], r["variant"], r["repeat"], r["step"]) for r in artifact["rows"]]
            require(
                len(keys) == len(set(keys)) and set(keys) == expected, f"Estimator rows: {path}"
            )
            require(len(artifact["costs"]) == len(variants), f"Workload count: {path}")
            for row in artifact["rows"]:
                require(
                    row["observation_sha256"] == artifact["observation_sha256"], f"Pairing: {path}"
                )
                require(np.isfinite(row["theta_hat"]), f"Finite estimate: {path}")
                close(row["error"], row["theta_hat"] - case["theta_true"], "signed error")
                close(row["absolute_error"], abs(row["error"]), "absolute error")
            # Independent verification of maxima and score signs against saved surfaces.
            for surface in artifact["surfaces"]:
                values = np.asarray(surface["values"])
                grid = np.asarray(surface["theta_grid"])
                require(
                    values.shape == (len(grid), len(manifest["checkpoint_steps"])), "Surface shape"
                )
                selected = [
                    row
                    for row in artifact["rows"]
                    if row["method"] == surface["method"]
                    and row["variant"] == surface["variant"]
                    and row["repeat"] == surface["repeat"]
                ]
                if surface["kind"] == "particle_log_likelihood":
                    for index, row in enumerate(selected):
                        close(
                            row["theta_hat"], grid[np.argmax(values[:, index])], "Particle argmax"
                        )
                if surface["kind"] == "augmented_EnKF_score":
                    for index, row in enumerate(selected):
                        if row["status"] in {"interpolated_root", "exact_root"}:
                            close(
                                np.interp(row["theta_hat"], grid, values[:, index]), 0, "Score root"
                            )
            artifacts.append(artifact)
            expected_rows += len(expected)
            expected_costs += len(variants)
    require(
        len({a["observation_sha256"] for a in artifacts}) == len(artifacts),
        "Duplicate observation records",
    )
    files = list((root / "records").glob("*/*.json"))
    require(len(files) == len(artifacts), "Unexpected extra record files")
    estimates, costs, summaries = (
        read_csv(root / name) for name in ("estimates.csv", "costs.csv", "summary.csv")
    )
    require(
        canonical(estimates) == canonical([r for a in artifacts for r in a["rows"]]),
        "Estimates CSV differs from individual records",
    )
    require(
        canonical(costs) == canonical([r for a in artifacts for r in a["costs"]]),
        "Costs CSV differs from individual records",
    )
    expected_summary_count = (
        2 * len(manifest["experiment"]["cases"]) * len(manifest["checkpoint_steps"])
    )
    require(len(summaries) == expected_summary_count, "Summary cardinality")
    lookup = {(row["case"], row["method"], int(row["step"])): row for row in summaries}
    require(len(lookup) == expected_summary_count, "Duplicate summaries")
    family = len(manifest["experiment"]["cases"]) * len(manifest["checkpoint_steps"])
    for case in manifest["experiment"]["cases"]:
        baseline = "kalman" if case["name"] == "ou_linear_gaussian" else "particle"
        for step in manifest["checkpoint_steps"]:
            errors = {}
            for method in ("score_root", baseline):
                group = sorted(
                    [
                        r
                        for r in estimates
                        if r["case"] == case["name"]
                        and r["method"] == method
                        and r["variant"] == "primary"
                        and int(r["step"]) == step
                    ],
                    key=lambda r: int(r["replicate"]),
                )
                require(
                    [int(r["replicate"]) for r in group] == list(range(case["replicates"])),
                    "Complete primary replicates",
                )
                error = np.array([float(r["error"]) for r in group])
                errors[method] = error
                row = lookup[(case["name"], method, step)]
                for key, expected in {
                    "mae": abs(error).mean(),
                    "rmse": np.sqrt(np.mean(error**2)),
                    "bias": error.mean(),
                    "estimate_sd": error.std(ddof=1),
                }.items():
                    close(float(row[key]), expected, key)
                boundary = np.mean(
                    [
                        np.isclose(float(r["theta_hat"]), case["theta_min"], rtol=0, atol=1e-8)
                        or np.isclose(float(r["theta_hat"]), case["theta_max"], rtol=0, atol=1e-8)
                        for r in group
                    ]
                )
                close(float(row["boundary_rate"]), boundary, "boundary rate")
                if method == "score_root":
                    missing = np.mean(
                        [r["status"] not in {"exact_root", "interpolated_root"} for r in group]
                    )
                    close(float(row["missing_root_rate"]), missing, "missing root rate")
            diff = abs(errors["score_root"]) - abs(errors[baseline])
            rng = seeded_rng(
                manifest["experiment"]["master_seed"], case["name"], "comparison_bootstrap", step
            )
            means = np.concatenate(
                [
                    diff[
                        rng.integers(
                            0,
                            len(diff),
                            size=(min(1000, manifest["bootstrap_samples"] - start), len(diff)),
                        )
                    ].mean(axis=1)
                    for start in range(0, manifest["bootstrap_samples"], 1000)
                ]
            )
            row = lookup[(case["name"], "score_root", step)]
            expected = np.quantile(means, [0.025, 0.975, 0.025 / family, 1 - 0.025 / family])
            for key, value in zip(
                ("paired_ci_low", "paired_ci_high", "family_ci_low", "family_ci_high"),
                expected,
                strict=True,
            ):
                close(float(row[key]), float(value), key)
            close(float(row["delta_mae_vs_baseline"]), diff.mean(), "paired mean difference")
            close(
                float(row["paired_delta_mcse"]),
                diff.std(ddof=1) / np.sqrt(len(diff)),
                "paired MCSE",
            )
    audit_result = {
        "record_files": len(artifacts),
        "estimate_rows": expected_rows,
        "cost_rows": expected_costs,
        "summary_rows": expected_summary_count,
        "status": "passed",
        "run_source_sha256": manifest["source_sha256"],
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_file_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("manifest.json", "estimates.csv", "costs.csv", "summary.csv")
        },
    }
    return manifest, estimates, costs, summaries, audit_result


def analyze(root: Path, output: Path) -> None:
    # The prose interpretation below is specific to this completed study.
    # Reject another protocol instead of attaching its results to that narrative.
    require(
        hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        == "2878a513601790f684ae123eb5b88668973c72f0accd6f0e7616b9f4805470bf",
        "This report is specific to the completed 20261008 study; manifest differs",
    )
    manifest, rows, costs, summaries, audit_result = audit(root)
    output.mkdir(parents=True, exist_ok=True)
    cases = manifest["experiment"]["cases"]
    final = manifest["checkpoint_steps"][-1]
    lookup = {(r["case"], r["method"], int(r["step"])): r for r in summaries}
    final_rows, numerical, projections = [], [], []
    for case in cases:
        name = case["name"]
        baseline = "kalman" if name == "ou_linear_gaussian" else "particle"
        score, base = (lookup[(name, m, final)] for m in ("score_root", baseline))
        medians = {
            m: float(
                np.median(
                    [
                        float(r["cpu_s"])
                        for r in costs
                        if r["case"] == name and r["method"] == m and r["variant"] == "primary"
                    ]
                )
            )
            for m in ("score_root", baseline)
        }
        final_rows.append(
            {
                "case": name,
                "baseline": baseline,
                "replicates": case["replicates"],
                "score_mae": float(score["mae"]),
                "baseline_mae": float(base["mae"]),
                "delta_mae": float(score["delta_mae_vs_baseline"]),
                "family_ci_low": float(score["family_ci_low"]),
                "family_ci_high": float(score["family_ci_high"]),
                "score_rmse": float(score["rmse"]),
                "baseline_rmse": float(base["rmse"]),
                "score_bias": float(score["bias"]),
                "baseline_bias": float(base["bias"]),
                "relative_mae_reduction_pct": 100 * (1 - float(score["mae"]) / float(base["mae"])),
                "score_median_cpu_s": medians["score_root"],
                "baseline_median_cpu_s": medians[baseline],
            }
        )
        for method in ("score_root", baseline):
            for variant in ("seed_repeat", "particle_count", "grid_resolution"):
                group = [
                    r
                    for r in rows
                    if r["case"] == name
                    and r["method"] == method
                    and r["variant"] == variant
                    and int(r["step"]) == final
                ]
                if not group:
                    continue
                primary = {
                    int(r["replicate"]): r
                    for r in rows
                    if r["case"] == name
                    and r["method"] == method
                    and r["variant"] == "primary"
                    and int(r["step"]) == final
                }
                shifts = np.array(
                    [
                        abs(
                            float(r["theta_hat"]) - float(primary[int(r["replicate"])]["theta_hat"])
                        )
                        for r in group
                    ]
                )
                numerical.append(
                    {
                        "case": name,
                        "method": method,
                        "variant": variant,
                        "records": len(group),
                        "median_absolute_shift": float(np.median(shifts)),
                        "maximum_absolute_shift": float(shifts.max()),
                        "primary_subset_mae": float(
                            np.mean(
                                [
                                    float(primary[int(r["replicate"])]["absolute_error"])
                                    for r in group
                                ]
                            )
                        ),
                        "check_subset_mae": float(
                            np.mean([float(r["absolute_error"]) for r in group])
                        ),
                    }
                )
        if baseline == "particle":
            group = [
                r
                for r in rows
                if r["case"] == name
                and r["method"] == "score_root"
                and r["variant"] == "primary"
                and int(r["step"]) == final
            ]
            theta = np.array([float(r["theta_hat"]) for r in group])
            grid = np.linspace(case["theta_min"], case["theta_max"], case["theta_points"])
            rounded = grid[np.abs(theta[:, None] - grid).argmin(axis=1)]
            projected = float(np.mean(abs(rounded - case["theta_true"])))
            projections.append(
                {
                    "case": name,
                    "diagnostic": "post_hoc_score_projection_to_primary_particle_grid",
                    "score_original_mae": float(score["mae"]),
                    "score_projected_mae": projected,
                    "particle_mae": float(base["mae"]),
                    "projected_delta_mae": projected - float(base["mae"]),
                    "minimum_possible_primary_grid_absolute_error": float(
                        np.min(abs(grid - case["theta_true"]))
                    ),
                }
            )
    write_csv(output / "final_comparison.csv", final_rows)
    write_csv(output / "numerical_sensitivity.csv", numerical)
    write_csv(output / "grid_projection_exploratory.csv", projections)
    (output / "audit.json").write_text(json.dumps(audit_result, indent=2) + "\n")
    figures(output, cases, final_rows, lookup, manifest)
    report(root, output, final_rows, numerical, projections, audit_result)
    print(json.dumps(audit_result, indent=2))
    print(f"Report: {output / 'report.md'}")


def figures(output: Path, cases: list, final_rows: list, lookup: dict, manifest: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    blue, orange, ink = "#166b98", "#b65b22", "#23354d"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    y = np.arange(len(cases))
    axes[0].barh(
        y - 0.17,
        [r["score_mae"] for r in final_rows],
        0.30,
        label="Augmented-EnKF score root",
        color=blue,
    )
    axes[0].barh(
        y + 0.17,
        [r["baseline_mae"] for r in final_rows],
        0.30,
        label="Kalman / particle grid likelihood",
        color=orange,
    )
    for i, r in enumerate(final_rows):
        axes[0].text(
            r["score_mae"] + 0.0006, i - 0.17, f"{r['score_mae']:.4f}", va="center", fontsize=9
        )
        axes[0].text(
            r["baseline_mae"] + 0.0006,
            i + 0.17,
            f"{r['baseline_mae']:.4f}",
            va="center",
            fontsize=9,
        )
    axes[0].set(
        yticks=y,
        yticklabels=[LABELS[c["name"]] for c in cases],
        xlim=(0, 0.047),
        xlabel="Mean absolute parameter error (lower is better)",
        title="Accuracy at T = 1000",
    )
    axes[0].set_ylim(3.2, -0.6)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    means = np.array([r["delta_mae"] for r in final_rows])
    lows = np.array([r["family_ci_low"] for r in final_rows])
    highs = np.array([r["family_ci_high"] for r in final_rows])
    axes[1].errorbar(
        means, y, xerr=[means - lows, highs - means], fmt="o", color=ink, capsize=5, linewidth=2
    )
    axes[1].axvline(0, color="#6e7c8d", linestyle="--", linewidth=1)
    axes[1].set(
        yticks=y,
        yticklabels=[LABELS[c["name"]] for c in cases],
        xlim=(-0.031, 0.015),
        xlabel="MAE(score root) − MAE(baseline)",
        title="Paired differences with adjusted intervals",
    )
    axes[1].set_ylim(3.2, -0.6)
    axes[1].text(
        -0.015,
        0.02,
        "← Favors score root",
        transform=axes[1].get_xaxis_transform(),
        ha="center",
        fontsize=8,
    )
    axes[1].text(
        0.0075,
        0.02,
        "Favors baseline →",
        transform=axes[1].get_xaxis_transform(),
        ha="center",
        fontsize=8,
    )
    fig.suptitle(
        "Paired OU parameter estimation · 40 independent records per model", fontsize=13, color=ink
    )
    fig.supxlabel(
        "Intervals: approximate 95% family coverage across 9 comparisons. "
        "Particle estimates use a 0.05-spaced grid; score roots interpolate.",
        fontsize=8,
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"final_comparison.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), sharey=True, layout="constrained")
    for ax, case in zip(axes, cases, strict=True):
        name = case["name"]
        baseline = "kalman" if name == "ou_linear_gaussian" else "particle"
        times = np.array(manifest["checkpoint_steps"]) * case["dt"]
        for method, color, label in (
            ("score_root", blue, "Score root"),
            (baseline, orange, "Kalman" if baseline == "kalman" else "Particle grid MLE"),
        ):
            ax.plot(
                times,
                [
                    float(lookup[(name, method, step)]["mae"])
                    for step in manifest["checkpoint_steps"]
                ],
                "o-",
                color=color,
                label=label,
            )
        ax.set(title=LABELS[name], xlabel="Observation horizon", xticks=times, ylim=(0, 0.08))
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Mean absolute parameter error")
    fig.suptitle("Accuracy as the observation horizon increases", fontsize=13, color=ink)
    fig.supxlabel(
        "Same 40 record prefixes at each checkpoint. Connecting lines guide the "
        "eye; they do not establish an asymptotic rate.",
        fontsize=8,
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"horizon_comparison.{suffix}", dpi=180)
    plt.close(fig)


def report(
    root: Path, output: Path, finals: list, sensitivity: list, projection: list, audit_result: dict
) -> None:
    lines = [
        "# Completed paired OU parameter-estimation comparison",
        "",
        "Forty independent observation records per model, true parameter 0.37, Euler step 0.01, "
        "and checkpoints T=250, 500, 1000. Methods use identical observations within each record. "
        "The final horizon and paired MAE were the primary comparison; earlier "
        "checkpoints were secondary.",
        "",
        "## Final accuracy",
        "",
        "The interval columns use the planned Bonferroni-adjusted percentile bootstrap: "
        "50,000 record resamples, approximate 95% family coverage over all nine "
        "model/checkpoint pairs. "
        "Negative differences favor the score root. Checkpoints are dependent "
        "observations from the same records; "
        "they are not extra independent replicates.",
        "",
        "| Model / baseline | Score MAE | Baseline MAE | Paired MAE difference | "
        "Adjusted interval |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for r in finals:
        lines.append(
            f"| {LABELS[r['case']]} / {r['baseline']} | {r['score_mae']:.4f} | "
            f"{r['baseline_mae']:.4f} | "
            f"{r['delta_mae']:+.4f} | [{r['family_ci_low']:+.4f}, {r['family_ci_high']:+.4f}] |"
        )
    lines += [
        "",
        "The score root has lower MAE than the implemented particle-grid "
        "estimator in both nonlinear "
        "models at T=1000, with intervals excluding zero after adjustment. Its observed MAE is "
        "31.4% lower for nonlinear Gaussian and 44.9% lower for Poisson observations. "
        "These percentages describe point estimates for the implemented configurations. "
        "Kalman has lower point MAE in the linear model; the interval includes "
        "zero, so the accuracy "
        "difference is unresolved by these 40 records. It does not demonstrate equivalence.",
        "",
        "All methods improve in MAE between T=250 and T=1000. Only the two final nonlinear "
        "comparisons exclude zero under the adjustment across all nine comparisons. "
        "No primary estimate reached a parameter boundary, and all 360 score estimates at the "
        "reported checkpoints had a root. This does not assert root availability "
        "at every intermediate step.",
        "",
        "| Model | RMSE score / baseline | Bias score / baseline |",
        "| --- | ---: | ---: |",
    ]
    for r in finals:
        lines.append(
            f"| {LABELS[r['case']]} | {r['score_rmse']:.4f} / {r['baseline_rmse']:.4f} | "
            f"{r['score_bias']:+.4f} / {r['baseline_bias']:+.4f} |"
        )
    lines += [
        "",
        "Poisson score-root bias is larger than the particle baseline's bias despite lower MAE "
        "and RMSE; its smaller spread improves the overall error. A ranking by bias alone would "
        "therefore tell a different and incomplete story.",
        "",
        "## Numerical resolution and stability",
        "",
        "The primary particle grid is spaced by 0.05 and its closest value to 0.37 is 0.35. "
        "Every primary particle estimate therefore has absolute error at least 0.02, even if its "
        "likelihood were evaluated perfectly. The score estimator interpolates, and the Kalman "
        "optimizer is continuous. This matters particularly for the Poisson score MAE of 0.0184, "
        "which is below that grid-imposed floor.",
        "",
        "As an explicitly **post hoc descriptive diagnostic**, project each score estimate to the "
        "nearest point on the same primary particle grid. This requires no new "
        "filtering and changes "
        "only the score estimator's output resolution. It is not a replacement "
        "for the planned analysis.",
        "",
        "| Model | Score MAE, original | Score MAE, projected to particle grid | Particle MAE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in projection:
        lines.append(
            f"| {LABELS[r['case']]} | {r['score_original_mae']:.4f} | "
            f"{r['score_projected_mae']:.4f} | {r['particle_mae']:.4f} |"
        )
    lines += [
        "",
        "The remaining descriptive gaps are 0.0075 and 0.0060. Projection shrinks the original "
        "gaps by approximately 37% and 60%, respectively. It does not isolate "
        "all numerical effects "
        "or establish what a continuously optimized particle likelihood would achieve.",
        "",
        "The numerical checks use the fixed first five records per model and are not pooled "
        "with the primary 40. The table reports absolute changes in the "
        "parameter estimate at T=1000.",
        "",
        "| Model | Method / check | Median change | Maximum change |",
        "| --- | --- | ---: | ---: |",
    ]
    for r in sensitivity:
        lines.append(
            f"| {LABELS[r['case']]} | {r['method']} / {r['variant']} | "
            f"{r['median_absolute_shift']:.4f} | {r['maximum_absolute_shift']:.4f} |"
        )
    lines += [
        "",
        "Particle-seed changes reach 0.10 for nonlinear Gaussian and 0.05 for Poisson; "
        "doubling particles changes estimates by up to 0.05 in each. The finer 37-point grid "
        "changes estimates by up to 0.025. EnKF-seed changes are also material: maxima "
        "0.0412, 0.0302, and 0.0172 for linear Gaussian, nonlinear Gaussian, and Poisson. "
        "Five paired checks cannot estimate numerical convergence or decompose error reliably. "
        "These changes are comparable to, or larger than, the mean accuracy gaps, so "
        "claims concern the tested randomized implementations rather than "
        "infinite-ensemble estimators.",
        "",
        "## Update schedule, resources, and computation",
        "",
        "| Method | State representation | Parameter treatment and update schedule |",
        "| --- | --- | --- |",
        "| Score root | 75 augmented members per candidate for linear Gaussian, "
        "150 for other models | "
        "19 fixed candidates; score updates at every observation; interpolated "
        "roots define a running estimate. "
        "The runner computes candidate paths in batches and evaluates the full "
        "root continuation. |",
        "| Kalman likelihood | Gaussian mean and variance, no state ensemble | Continuous bounded "
        "parameter optimization at each checkpoint; each likelihood evaluation "
        "refilters the available prefix. |",
        "| Particle likelihood | 8,192 hidden-state particles per candidate | 19 fixed candidates; "
        "update cumulative likelihood and state particles at every observation, "
        "take the grid maximum "
        "at checkpoints. The bank could return a grid maximum every step without refiltering. |",
        "",
        "Fixed parameter banks are not learned parameter-posterior ensembles. The particle bank "
        "contains 155,648 hidden-state particles; the nonlinear score bank has 2,850 augmented "
        "ensemble members. Their members carry different quantities, so these are counts, not "
        "equivalent units of memory or computation. All methods use filtering information only, "
        "with no future observations or smoothing.",
        "",
        "| Model | Median primary CPU seconds: score | Median primary CPU seconds: baseline |",
        "| --- | ---: | ---: |",
    ]
    for r in finals:
        lines.append(
            f"| {LABELS[r['case']]} | {r['score_median_cpu_s']:.1f} | "
            f"{r['baseline_median_cpu_s']:.2f} |"
        )
    lines += [
        "",
        "For these implementations, Kalman uses about 151 times less CPU than the score-root "
        "workload in the linear model. The score-root workload uses about 2.31 and 2.08 times less "
        "CPU than the particle workloads in the two nonlinear models. Costs "
        "exclude data simulation, "
        "artifact writing, and numerical checks. They include the full running root path, three "
        "Kalman prefix optimizations, or one cumulative particle likelihood bank. Timing came "
        "from parallel launches with varying load; these are workload measurements, not a "
        "controlled latency benchmark or equal-budget optimization study.",
        "",
        "## Interpretation",
        "",
        "The score-root method demonstrates accurate running parameter estimation in these "
        "partially observed models. At the final horizon it combines lower error and lower "
        "measured computation than the tested particle-grid likelihood implementation. "
        "The exact linear-Gaussian likelihood remains the practical reference "
        "when its assumptions hold. "
        "The nonlinear results identify a useful accuracy/computation tradeoff for the implemented "
        "settings; resolution and seed sensitivity prevent a broader claim of superiority over "
        "particle-based parameter inference.",
        "",
        "The 40-record bootstrap quantifies sampling variation for these randomized algorithms "
        "under the simulated model. It does not remove grid bias, EnKF approximation error, or "
        "particle likelihood error, and it is not a confidence interval for a "
        "single record's parameter. "
        "One true parameter, one noise setting per model, and one Euler step are evaluated. "
        "The three horizons do not establish an asymptotic rate. Algorithmic improvements should "
        "be evaluated on fresh records with a declared computational budget.",
        "",
        "## Method references",
        "",
        "- Kalman, R. E. (1960), *A New Approach to Linear Filtering and Prediction Problems*. "
        "[Original "
        "paper](https://people.math.harvard.edu/archive/116_fall_03/handouts/Kalman1960.pdf). "
        "This is the state-filter foundation; the implemented estimator adds "
        "innovation-likelihood optimization.",
        "- Gordon, N. J., Salmond, D. J., and Smith, A. F. M. (1993), "
        "*Novel approach to nonlinear/non-Gaussian Bayesian state estimation*. "
        "[Original bootstrap-filter "
        "paper](https://people.bordeaux.inria.fr/pierre.delmoral/gordon-salmond-smith-1993.pdf). "
        "The implementation uses adaptive systematic resampling and maximizes "
        "its likelihood estimate on a grid.",
        "- Kantas et al. (2015), *On Particle Methods for Parameter Estimation "
        "in State-Space Models*. "
        "[Review](https://arxiv.org/abs/1412.8695). Places particle likelihood estimation and "
        "static-parameter inference in their wider methodological context.",
        "- Evensen (2003), *The Ensemble Kalman Filter: theoretical formulation "
        "and practical implementation*. "
        "[Author paper via "
        "ECMWF](https://www.ecmwf.int/sites/default/files/elibrary/2003/7442"
        "4-ensemble-kalman-filter-theoretical-formulation-and-practical-implementation_0.pdf). "
        "Ensemble-filter background, rather than a source for this thesis's "
        "score-root construction.",
        "- Chopin, Jacob, and Papaspiliopoulos, *SMC²: an efficient algorithm "
        "for sequential analysis "
        "of state-space models*. [Paper](https://arxiv.org/abs/1101.1528). "
        "An unbenchmarked sequential Bayesian alternative with parameter "
        "particles and nested state filters; "
        "its posterior target differs from this point-estimation study.",
        "",
        "## Audit and regeneration",
        "",
        f"Verified {audit_result['record_files']} record files, "
        f"{audit_result['estimate_rows']} estimate rows, "
        f"{audit_result['cost_rows']} cost rows, and {audit_result['summary_rows']} summary rows. "
        "The audit checks complete planned variants, pairing checksums, finite estimates, "
        "saved particle maxima and score roots, every accuracy metric, and all "
        "bootstrap intervals. "
        "The original run artifacts remain unchanged. Input hashes are in `audit.json`.",
        "",
        "```bash",
        f"python scripts/analyze_comparison.py --input {shlex.quote(str(root))}",
        "```",
        "",
        "![Final comparison](final_comparison.png)",
        "",
        "![Horizons](horizon_comparison.png)",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/comparison-full"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analyze(args.input, args.output or args.input / "analysis")
