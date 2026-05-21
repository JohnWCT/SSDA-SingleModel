"""Ensure validation phase does not call optimizer.step."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import model as m
from ssda_latent import safe_trainer


def _loader_dict(n: int = 8, feat: int = 4) -> dict[str, DataLoader]:
    x = torch.randn(n, feat)
    y = torch.randint(0, 2, (n,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    return {"train": loader, "val": loader}


def test_val_phase_does_not_step_optimizer() -> None:
    encoder = m.MLP(input_dim=4, latent_dim=4, h_dims=[8], drop_out=0.0)
    predictor = m.Predictor(input_dim=4, output_dim=32, drop_out=0.0)
    adentropy_p = m.Predictor_adentropy(num_class=2, inc=32)
    optimizer = torch.optim.SGD(
        list(encoder.parameters())
        + list(predictor.parameters())
        + list(adentropy_p.parameters()),
        lr=0.01,
    )
    step_calls: list[str] = []
    original_step = optimizer.step

    def counting_step(*args, **kwargs):
        step_calls.append("step")
        return original_step(*args, **kwargs)

    optimizer.step = counting_step  # type: ignore[method-assign]

    loaders = _loader_dict()
    safe_trainer.train_semi_mlp(
        encoder,
        predictor,
        adentropy_p,
        loaders,
        loaders,
        loaders,
        "adv",
        optimizer,
        nn.CrossEntropyLoss(),
        nn.MSELoss(),
        n_epochs=1,
        device="cpu",
        auc_path="/tmp/ssda_safe_trainer_test",
    )
    # n=8, batch_size=4 -> 2 train batches; each batch: supervised + MME = 4 steps
    # validation phase must not call optimizer.step
    assert len(step_calls) == 4
