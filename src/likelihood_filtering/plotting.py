from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from .persistence import read_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_figure(figure: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_experiment(output_dir: str | Path) -> list[Path]:
    """Create estimator and information figures from saved artifacts."""
    output_dir = Path(output_dir)
    manifest = read_json(output_dir / "manifest.json")
    written: list[Path] = []
    for case in manifest["experiment"]["cases"]:
        case_name = str(case["name"])
        theta_true = float(case["theta_true"])
        case_dir = output_dir / case_name
        figure_dir = case_dir / "figures"
        with np.load(case_dir / "aggregates.npz") as aggregate:
            time = aggregate["time"]
            quantiles = aggregate["theta_quantiles"]
            figure, axis = plt.subplots(figsize=(5.2, 3.2))
            axis.fill_between(time, quantiles[0], quantiles[2], alpha=0.2)
            axis.plot(time, quantiles[1], label="median estimate")
            axis.axhline(theta_true, color="black", linestyle="--", label="true value")
            axis.set(xlabel="time", ylabel=r"$\widehat{\theta}$")
            axis.legend(frameon=False)
            _save_figure(figure, figure_dir, "estimate")
            written.extend([figure_dir / "estimate.pdf", figure_dir / "estimate.png"])

            figure, axis = plt.subplots(figsize=(5.2, 3.2))
            axis.plot(time, aggregate["observed_information_mean"], label="Louis")
            axis.plot(time, aggregate["predictable_information_mean"], label="score QV")
            axis.set(xlabel="time", ylabel="information")
            axis.legend(frameon=False)
            _save_figure(figure, figure_dir, "information")
            written.extend([figure_dir / "information.pdf", figure_dir / "information.png"])

        replicate_path = case_dir / "replicates" / "replicate_0000.npz"
        with np.load(replicate_path, allow_pickle=False) as replicate:
            if "score_surface" in replicate:
                theta_grid = replicate["theta_grid"]
                score_surface = replicate["score_surface"]
                times = replicate["time"]
                indices = np.unique(
                    np.linspace(1, times.size - 1, min(4, times.size - 1), dtype=int)
                )
                figure, axis = plt.subplots(figsize=(5.2, 3.2))
                for index in indices:
                    axis.plot(theta_grid, score_surface[:, index], label=f"t={times[index]:g}")
                axis.axhline(0.0, color="black", linewidth=0.8)
                axis.set(xlabel=r"$\theta$", ylabel="estimated score")
                axis.legend(frameon=False)
                _save_figure(figure, figure_dir, "score_roots")
                written.extend([figure_dir / "score_roots.pdf", figure_dir / "score_roots.png"])
    return written
