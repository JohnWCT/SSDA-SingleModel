"""SSDA4Drug latent extension entry point (does not modify experiment_shot.py)."""

from __future__ import annotations

import warnings

import matplotlib

from ssda_latent.config import build_arg_parser, config_from_args
from ssda_latent.orchestrator import ExperimentRunner
from ssda_latent.seed import SeedManager


def main() -> None:
    parser = build_arg_parser()
    args, _unknown = parser.parse_known_args()
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore")
    config = config_from_args(args)
    SeedManager.set_all(config.random_seed)
    ExperimentRunner(config).run()


if __name__ == "__main__":
    main()
