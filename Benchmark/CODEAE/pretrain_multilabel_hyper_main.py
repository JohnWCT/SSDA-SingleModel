#!/usr/bin/env python3
"""Thin entry: CODE-AE multilabel pretraining."""

from __future__ import annotations

import sys
from pathlib import Path

_CODEAE_ROOT = Path(__file__).resolve().parent
if str(_CODEAE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODEAE_ROOT))

from codeae_multilabel.config import build_pretrain_arg_parser, config_from_pretrain_args
from codeae_multilabel.seed import set_global_seed
from codeae_multilabel.training.runners import PretrainRunner


def main() -> None:
    parser = build_pretrain_arg_parser()
    args = parser.parse_args()
    config = config_from_pretrain_args(args)
    set_global_seed(config.seed)
    PretrainRunner(config).run()


if __name__ == "__main__":
    main()
