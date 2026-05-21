"""Train SSDA models per fold using legacy trainer."""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

import model as m
from ssda_latent import safe_trainer
import utils
from ssda_latent.config import ExperimentConfig
from ssda_latent.dataloader_factory import FoldDataLoaders


@dataclass(frozen=True)
class ModelBundle:
    encoder: nn.Module
    predictor: nn.Module
    adentropy_p: nn.Module
    fgm: m.FGM | None
    device: torch.device


@dataclass(frozen=True)
class TrainingResult:
    fold_index: int
    encoder: nn.Module
    predictor: nn.Module
    adentropy_p: nn.Module
    checkpoint_path: str
    legacy_auc_path: str


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "gpu":
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(dev)
        return dev
    return torch.device("cpu")


def build_para_string(config: ExperimentConfig) -> str:
    return (
        f"_drug_{config.drug}_method_{config.method}_gene_{config.gene}"
        f"_DAEdim_{','.join(map(str, config.encoder_h_dims))}"
        f"_Predim_{','.join(map(str, config.predictor_h_dims))}"
        f"_dropout_{config.dropout}_lr_{config.lr}_bs_{config.batch_size}"
    )


def build_models(config: ExperimentConfig, device: torch.device) -> ModelBundle:
    bulk_tasks, sc_tasks = utils.cell_dim(drug=config.drug, gene=config.gene)
    latent_dim = config.latent_dim if config.latent_dim is not None else sc_tasks["pathway"]
    encoder: nn.Module
    if config.encoder == "DAE":
        encoder = m.DAE(
            input_dim=latent_dim,
            fc_dim=256,
            AE_input_dim=bulk_tasks["expression"],
            AE_h_dims=list(config.encoder_h_dims),
            pretrained_weights=None,
            drop=config.dropout,
        )
    elif config.encoder == "MLP":
        encoder = m.MLP(
            input_dim=sc_tasks["expression"],
            latent_dim=latent_dim,
            h_dims=list(config.encoder_h_dims),
            drop_out=config.dropout,
        )
    else:
        raise ValueError("encoder must be DAE or MLP")
    encoder.to(device)
    predictor = m.Predictor(input_dim=latent_dim, output_dim=32, drop_out=config.dropout)
    predictor.to(device)
    adentropy_p = m.Predictor_adentropy(num_class=2, inc=32)
    adentropy_p.to(device)
    fgm = m.FGM([encoder, predictor, adentropy_p]) if config.encoder == "DAE" else None
    return ModelBundle(
        encoder=encoder,
        predictor=predictor,
        adentropy_p=adentropy_p,
        fgm=fgm,
        device=device,
    )


def ensure_legacy_dirs(config: ExperimentConfig) -> tuple[str, str]:
    for path in [
        config.umap_path + config.drug,
        config.sc_all + config.drug,
        config.result + "/" + config.method,
    ]:
        os.makedirs(path, exist_ok=True)
    para = build_para_string(config)
    sc_all_path = config.sc_all + config.drug + "/" + para
    result_path = config.result + "/" + config.method + "/" + config.drug + "/"
    os.makedirs(result_path, exist_ok=True)
    return sc_all_path, result_path


def train_fold(
    config: ExperimentConfig,
    loaders: FoldDataLoaders,
    fold_index: int,
    layout_fold_dir: Path,
) -> TrainingResult:
    device = resolve_device(config.device)
    bundle = build_models(config, device)
    sc_all_path, result_path = ensure_legacy_dirs(config)
    loss_c = nn.CrossEntropyLoss()
    loss_e = nn.MSELoss()
    optimizer = torch.optim.Adagrad(
        itertools.chain(
            bundle.encoder.parameters(),
            bundle.predictor.parameters(),
            bundle.adentropy_p.parameters(),
        ),
        lr=config.lr,
    )
    if config.encoder == "DAE":
        assert bundle.fgm is not None
        encoder_f, predictor_f, adentropy_f = safe_trainer.train_semi_dae(
            bundle.fgm,
            bundle.encoder,
            bundle.predictor,
            bundle.adentropy_p,
            loaders.source,
            loaders.target_unlabeled,
            loaders.target_labeled,
            config.method,
            optimizer,
            loss_c,
            loss_e,
            config.epochs,
            start_epoch=0,
            save_path=sc_all_path + ".pkl",
            device=device,
            auc_path=result_path,
        )
    else:
        encoder_f, predictor_f, adentropy_f = safe_trainer.train_semi_mlp(
            bundle.encoder,
            bundle.predictor,
            bundle.adentropy_p,
            loaders.source,
            loaders.target_unlabeled,
            loaders.target_labeled,
            config.method,
            optimizer,
            loss_c,
            loss_e,
            config.epochs,
            start_epoch=0,
            save_path=sc_all_path + ".pkl",
            device=device,
            auc_path=result_path,
        )
    if config.save_legacy_outputs:
        ext = ".pth"
        torch.save(
            {
                "encoder_state_dict": encoder_f.state_dict(),
                "predictor_state_dict": predictor_f.state_dict(),
                "Predictor_adentropy_state_dict": adentropy_f.state_dict(),
            },
            sc_all_path + ext,
        )
    ckpt_path = str(layout_fold_dir / "model_final.pth")
    torch.save(
        {
            "encoder_state_dict": encoder_f.state_dict(),
            "predictor_state_dict": predictor_f.state_dict(),
            "Predictor_adentropy_state_dict": adentropy_f.state_dict(),
            "fold": fold_index,
            "seed": config.random_seed,
            "drug": config.drug,
        },
        ckpt_path,
    )
    encoder_f.eval()
    predictor_f.eval()
    adentropy_f.eval()
    return TrainingResult(
        fold_index=fold_index,
        encoder=encoder_f,
        predictor=predictor_f,
        adentropy_p=adentropy_f,
        checkpoint_path=ckpt_path,
        legacy_auc_path=result_path,
    )
