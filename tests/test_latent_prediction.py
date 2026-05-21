"""Tests for deterministic DAE latent and MLP prediction shapes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import model as m
from ssda_latent.latent import encode_latent_dict, get_encoder_latent
from ssda_latent.prediction import predict_dataframe


def test_dae_latent_is_deterministic() -> None:
    torch.manual_seed(0)
    dae = m.DAE(
        input_dim=8,
        fc_dim=16,
        AE_input_dim=12,
        AE_h_dims=[16],
        drop=0.0,
    )
    dae.eval()
    x = torch.randn(5, 12)
    z1 = get_encoder_latent(dae, x, deterministic=True)
    z2 = get_encoder_latent(dae, x, deterministic=True)
    assert torch.allclose(z1, z2)

    z_stochastic = get_encoder_latent(dae, x, deterministic=False)
    # Stochastic forward may differ (random mask); at least shape matches
    assert z_stochastic.shape == z1.shape


def test_mlp_prediction_full_batch() -> None:
    encoder = m.MLP(input_dim=10, latent_dim=8, h_dims=[16], drop_out=0.0)
    predictor = m.Predictor(input_dim=8, output_dim=32, drop_out=0.0)
    adentropy = m.Predictor_adentropy(num_class=2, inc=32)
    n = 7
    x_df = pd.DataFrame(
        np.random.default_rng(1).normal(size=(n, 10)),
        index=[f"s{i}" for i in range(n)],
    )
    out = predict_dataframe(
        encoder, predictor, adentropy, x_df, torch.device("cpu"), batch_size=4
    )
    assert len(out) == n
    assert out["probability_class_1"].between(0, 1).all()


def test_encode_latent_dict_keys() -> None:
    encoder = m.MLP(input_dim=6, latent_dim=4, h_dims=[8], drop_out=0.0)
    x_df = pd.DataFrame(
        np.ones((3, 6)),
        index=["a", "b", "c"],
    )
    lat = encode_latent_dict(encoder, x_df, torch.device("cpu"), batch_size=2)
    assert set(lat.keys()) == {"a", "b", "c"}
    assert len(lat["a"]) == 4
