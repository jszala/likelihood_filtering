import csv
from dataclasses import replace
from pathlib import Path

from likelihood_filtering.benchmark import run_benchmark
from likelihood_filtering.config import ExperimentConfig


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_paired_benchmark_smoke_and_resume(tmp_path: Path) -> None:
    config = ExperimentConfig.from_yaml("configs/benchmark.yaml")
    config = replace(
        config,
        name="test_smoke",
        cases=[replace(case, steps=5, replicates=1, ensemble_size=8) for case in config.cases],
    )
    summary = run_benchmark(config, tmp_path, particles=32, check_particles=64, check_replicates=1)
    first = _rows(tmp_path / "per_record.csv")
    assert len(first) == 6
    assert len(summary) == 6
    assert len(_rows(tmp_path / "particle_check.csv")) == 2
    for case in config.cases:
        case_rows = [row for row in first if row["case"] == case.name]
        assert {row["method"] for row in case_rows} == {
            "score_root",
            "kalman" if case.name == "ou_linear_gaussian" else "particle_32",
        }
        assert {row["replicate"] for row in case_rows} == {"0"}
    assert (
        run_benchmark(
            config, tmp_path, particles=32, check_particles=64, check_replicates=1, resume=True
        )
        == summary
    )
    assert _rows(tmp_path / "per_record.csv") == first

    parallel_dir = tmp_path / "parallel"
    run_benchmark(
        config, parallel_dir, particles=32, check_particles=64, check_replicates=1, workers=2
    )
    parallel = _rows(parallel_dir / "per_record.csv")
    for sequential_row, parallel_row in zip(first, parallel, strict=True):
        for field in ("case", "replicate", "method", "theta_hat", "status"):
            assert sequential_row[field] == parallel_row[field]
