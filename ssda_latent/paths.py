"""Output path layout for latent_ssda runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ssda_latent.config import ExperimentConfig


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> RunLayout:
        run_dir = Path(config.latent_output_dir) / config.drug / f"seed_{config.random_seed}"
        return cls(run_dir=run_dir)

    def fold_dir(self, fold_index: int) -> Path:
        return self.run_dir / f"fold_{fold_index}"

    def ensure_run_dir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def ensure_fold_dir(self, fold_index: int) -> Path:
        path = self.fold_dir(fold_index)
        path.mkdir(parents=True, exist_ok=True)
        return path
