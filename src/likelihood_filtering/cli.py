from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .experiment import run_experiment, select_cases, summarize_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="likelihood-filter")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run an experiment configuration")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--case", action="append", default=[])
    run.add_argument("--dry-run", action="store_true")

    summarize = commands.add_parser("summarize", help="rebuild compact summaries")
    summarize.add_argument("output", type=Path)

    plot = commands.add_parser("plot", help="plot saved experiment artifacts")
    plot.add_argument("output", type=Path)
    return parser


def _work_estimate(config: ExperimentConfig) -> int:
    total = 0
    for case in config.cases:
        filters = case.theta_points + 1 if case.estimator == "grid" else 1
        total += case.replicates * case.steps * case.ensemble_size * filters
    return total


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    args = _parser().parse_args(argv)
    if args.command == "run":
        config = ExperimentConfig.from_yaml(args.config)
        if args.case:
            config = select_cases(config, args.case)
        if args.dry_run:
            work = _work_estimate(config)
            print(f"{len(config.cases)} cases, approximately {work:,} ensemble steps")
            return 0
        run_experiment(config, args.output, workers=args.workers)
        from .plotting import plot_experiment

        plot_experiment(args.output)
    elif args.command == "summarize":
        summarize_experiment(args.output)
    else:
        from .plotting import plot_experiment

        plot_experiment(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
