"""Tests for config module."""

from __future__ import annotations

from ssda_latent.config import build_arg_parser, config_from_args


def test_config_from_args() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--drug", "Gefitinib", "--random_seed", "7", "--n_splits", "3"])
    cfg = config_from_args(args)
    assert cfg.drug == "Gefitinib"
    assert cfg.random_seed == 7
    assert cfg.n_splits == 3
    assert cfg.encoder_h_dims == (512, 256)
