"""Safe semi-supervised trainers: validation phase does not update weights.

Mirrors ``trainer.train_semi_dae`` / ``train_semi_mlp`` but only runs
``backward`` / ``optimizer.step`` when ``phase == "train"``.  Validation
uses ``torch.no_grad()`` forward passes for metrics only.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

import utils
from utils import save_AUCs


def train_semi_dae(
    fgm,
    encoder,
    predictor,
    adentropy_p,
    source_loader,
    unlabeled_loader,
    labeled_loader,
    method,
    optimizer,
    loss_class,
    loss_e,
    n_epochs,
    start_epoch=0,
    save_path="save/model.pkl",
    device="cuda",
    auc_path="",
):
    best_loss = np.inf
    best_auc = float("-inf")

    if not os.path.exists(auc_path):
        os.makedirs(auc_path)
    file_AUCs = auc_path + "/" + "_train_AUCs_" + ".txt"

    AUCs = "Epoch\tloss\tloss_dann\tloss_c\tloss_dae_s\tloss_dae_t\tphase\tauc\taupr"
    with open(file_AUCs, "w") as f:
        f.write(AUCs + "\n")

    for epoch in range(n_epochs):
        AUCtrain_source: list[float] = []
        AUCval_source: list[float] = []
        APRtrain_source: list[float] = []
        APRval_source: list[float] = []

        for phase in ["train", "val"]:
            if phase == "train":
                encoder.train()
                predictor.train()
                adentropy_p.train()
            else:
                encoder.eval()
                predictor.eval()
                adentropy_p.eval()

            running_loss_train: list[float] = []
            running_loss_valid: list[float] = []
            running_dann_train: list[float] = []
            running_dann_valid: list[float] = []
            running_c_train: list[float] = []
            running_c_valid: list[float] = []
            running_e_train_s: list[float] = []
            running_e_valid_s: list[float] = []
            running_e_train_t: list[float] = []
            running_e_valid_t: list[float] = []

            len_source = len(source_loader[phase])
            len_unlabeled = len(unlabeled_loader[phase])
            len_labeled = len(labeled_loader[phase])
            num_iter = max(len_source, len_unlabeled, len_labeled)

            for batch_idx in range(num_iter):
                if batch_idx % len_source == 0:
                    iter_source = iter(source_loader[phase])
                if batch_idx % len_unlabeled == 0:
                    iter_target_unl = iter(unlabeled_loader[phase])
                if batch_idx % len_labeled == 0:
                    iter_target = iter(labeled_loader[phase])
                xs, ys = iter_source.__next__()
                xt, yt = iter_target.__next__()
                xt_unl, _ = iter_target_unl.__next__()
                xt_unl = xt_unl.to(device)
                xs = xs.to(device)
                ys = ys.to(device)
                xt = xt.to(device)
                yt = yt.to(device)

                if phase == "train":
                    xs.requires_grad_(True)
                    xt.requires_grad_(True)
                    data = torch.cat((xs, xt), 0)
                    target = torch.cat((ys, yt), 0)

                    feature, ae_output = encoder(data)
                    output = predictor(feature)
                    output = adentropy_p(output)
                    loss_c = loss_class(output, target.long())
                    softmax_output = nn.Softmax(dim=1)(output)

                    loss_ae_t = loss_e(
                        xt, ae_output.narrow(0, xs.size(0), xt.size(0))
                    )
                    loss_ae_s = loss_e(xs, ae_output.narrow(0, 0, xs.size(0)))
                    loss = loss_c + loss_ae_t + loss_ae_s

                    loss.backward(retain_graph=True)
                    if method == "adv":
                        fgm.attack()
                        feature, ae_output = encoder(data)
                        output = predictor(feature)
                        output = adentropy_p(output)
                        loss_c_2 = loss_class(output, target.long())
                        loss_ae_t_2 = loss_e(
                            xt, ae_output.narrow(0, xs.size(0), xt.size(0))
                        )
                        loss_ae_s_2 = loss_e(
                            xs, ae_output.narrow(0, 0, xs.size(0))
                        )
                        loss_sum = loss_c_2 + loss_ae_t_2 + loss_ae_s_2
                        loss_sum.backward(retain_graph=True)
                        fgm.restore()
                    optimizer.step()
                    optimizer.zero_grad()

                    output_unl, _ = encoder(xt_unl)
                    output_unl = predictor(output_unl)
                    loss_t = utils.adentropy(adentropy_p, output_unl, 0.1)
                    loss_t.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    ys_cpu = ys.cpu()
                    y_pre = softmax_output.narrow(0, 0, xs.size(0)).cpu()[:, 1]
                    AUCtrain_source.append(
                        utils.roc_auc_score_trainval(
                            ys_cpu.detach().numpy(), y_pre.detach().numpy()
                        )
                    )
                    APRtrain_source.append(
                        average_precision_score(
                            ys_cpu.detach().numpy(), y_pre.detach().numpy()
                        )
                    )
                    running_loss_train.append(loss.item())
                    running_dann_train.append(-loss_t.item())
                    running_c_train.append(loss_c.item())
                    running_e_train_s.append(loss_ae_s.item())
                    running_e_train_t.append(loss_ae_t.item())
                else:
                    with torch.no_grad():
                        data = torch.cat((xs, xt), 0)
                        target = torch.cat((ys, yt), 0)
                        feature, ae_output = encoder(data)
                        output = predictor(feature)
                        output = adentropy_p(output)
                        loss_c = loss_class(output, target.long())
                        softmax_output = nn.Softmax(dim=1)(output)
                        loss_ae_t = loss_e(
                            xt, ae_output.narrow(0, xs.size(0), xt.size(0))
                        )
                        loss_ae_s = loss_e(xs, ae_output.narrow(0, 0, xs.size(0)))
                        loss = loss_c + loss_ae_t + loss_ae_s

                        output_unl, _ = encoder(xt_unl)
                        output_unl = predictor(output_unl)
                        loss_t = utils.adentropy(adentropy_p, output_unl, 0.1)

                        ys_cpu = ys.cpu()
                        y_pre = softmax_output.narrow(0, 0, xs.size(0)).cpu()[:, 1]
                        AUCval_source.append(
                            utils.roc_auc_score_trainval(
                                ys_cpu.detach().numpy(), y_pre.detach().numpy()
                            )
                        )
                        APRval_source.append(
                            average_precision_score(
                                ys_cpu.detach().numpy(), y_pre.detach().numpy()
                            )
                        )
                        running_loss_valid.append(loss.item())
                        running_dann_valid.append(-loss_t.item())
                        running_c_valid.append(loss_c.item())
                        running_e_valid_s.append(loss_ae_s.item())
                        running_e_valid_t.append(loss_ae_t.item())

            if phase == "train":
                epoch_loss = np.mean(np.array(running_loss_train))
                epoch_dann = np.mean(np.array(running_dann_train))
                epoch_c = np.mean(np.array(running_c_train))
                epoch_ae_s = np.mean(np.array(running_e_train_s))
                epoch_ae_t = np.mean(np.array(running_e_train_t))
                auc = np.mean(np.array(AUCtrain_source))
                aupr = np.mean(np.array(APRtrain_source))
            else:
                epoch_loss = np.mean(np.array(running_loss_valid))
                epoch_dann = np.mean(np.array(running_dann_valid))
                epoch_c = np.mean(np.array(running_c_valid))
                epoch_ae_s = np.mean(np.array(running_e_valid_s))
                epoch_ae_t = np.mean(np.array(running_e_valid_t))
                auc = np.mean(np.array(AUCval_source))
                aupr = np.mean(np.array(APRval_source))

            train_AUCs = [
                epoch,
                epoch_loss,
                epoch_dann,
                epoch_c,
                epoch_ae_s,
                epoch_ae_t,
                phase,
                auc,
                aupr,
            ]
            print(
                "Epoch:{} Phase:{} AUC: {:.3f} AUPR: {:.3f} ".format(
                    epoch, phase, auc, aupr
                )
            )
            save_AUCs(train_AUCs, file_AUCs)

            if (phase == "val") and (epoch_loss < best_loss or best_auc < auc):
                best_auc = auc
                best_loss = epoch_loss

    return encoder, predictor, adentropy_p


def train_semi_mlp(
    encoder,
    predictor,
    adentropy_p,
    source_loader,
    unlabeled_loader,
    labeled_loader,
    method,
    optimizer,
    loss_class,
    loss_e,
    n_epochs,
    start_epoch=0,
    save_path="save/model.pkl",
    device="cuda",
    auc_path="",
):
    del method, loss_e, start_epoch, save_path  # kept for API parity with legacy trainer

    best_loss = np.inf
    best_auc = float("-inf")

    if not os.path.exists(auc_path):
        os.makedirs(auc_path)
    file_AUCs = auc_path + "/" + "_train_AUCs_" + ".txt"

    AUCs = "Epoch\tloss\tloss_dann\tloss_c\tphase\tauc\taupr"
    with open(file_AUCs, "w") as f:
        f.write(AUCs + "\n")

    for epoch in range(n_epochs):
        AUCtrain_source: list[float] = []
        AUCval_source: list[float] = []
        APRtrain_source: list[float] = []
        APRval_source: list[float] = []

        for phase in ["train", "val"]:
            if phase == "train":
                encoder.train()
                predictor.train()
                adentropy_p.train()
            else:
                encoder.eval()
                predictor.eval()
                adentropy_p.eval()

            running_loss_train: list[float] = []
            running_loss_valid: list[float] = []
            running_dann_train: list[float] = []
            running_dann_valid: list[float] = []
            running_c_train: list[float] = []
            running_c_valid: list[float] = []

            len_source = len(source_loader[phase])
            len_unlabeled = len(unlabeled_loader[phase])
            len_labeled = len(labeled_loader[phase])
            num_iter = max(len_source, len_unlabeled, len_labeled)

            for batch_idx in range(num_iter):
                if batch_idx % len_source == 0:
                    iter_source = iter(source_loader[phase])
                if batch_idx % len_unlabeled == 0:
                    iter_target_unl = iter(unlabeled_loader[phase])
                if batch_idx % len_labeled == 0:
                    iter_target = iter(labeled_loader[phase])
                xs, ys = iter_source.__next__()
                xt, yt = iter_target.__next__()
                xt_unl, _ = iter_target_unl.__next__()
                xt_unl = xt_unl.to(device)
                xs = xs.to(device)
                ys = ys.to(device)
                xt = xt.to(device)
                yt = yt.to(device)

                if phase == "train":
                    xs.requires_grad_(True)
                    xt.requires_grad_(True)
                    data = torch.cat((xs, xt), 0)
                    target = torch.cat((ys, yt), 0)

                    feature = encoder(data)
                    output = predictor(feature)
                    output = adentropy_p(output)
                    loss_c = loss_class(output, target.long())
                    softmax_output = nn.Softmax(dim=1)(output)
                    loss = loss_c

                    loss.backward(retain_graph=True)
                    optimizer.step()
                    optimizer.zero_grad()

                    output_unl = encoder(xt_unl)
                    output_unl = predictor(output_unl)
                    loss_t = utils.adentropy(adentropy_p, output_unl, 0.1)
                    loss_t.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    ys_cpu = ys.cpu()
                    y_pre = softmax_output.narrow(0, 0, xs.size(0)).cpu()[:, 1]
                    AUCtrain_source.append(
                        utils.roc_auc_score_trainval(
                            ys_cpu.detach().numpy(), y_pre.detach().numpy()
                        )
                    )
                    APRtrain_source.append(
                        average_precision_score(
                            ys_cpu.detach().numpy(), y_pre.detach().numpy()
                        )
                    )
                    running_loss_train.append(loss.item())
                    running_dann_train.append(-loss_t.item())
                    running_c_train.append(loss_c.item())
                else:
                    with torch.no_grad():
                        data = torch.cat((xs, xt), 0)
                        target = torch.cat((ys, yt), 0)
                        feature = encoder(data)
                        output = predictor(feature)
                        output = adentropy_p(output)
                        loss_c = loss_class(output, target.long())
                        softmax_output = nn.Softmax(dim=1)(output)
                        loss = loss_c

                        output_unl = encoder(xt_unl)
                        output_unl = predictor(output_unl)
                        loss_t = utils.adentropy(adentropy_p, output_unl, 0.1)

                        ys_cpu = ys.cpu()
                        y_pre = softmax_output.narrow(0, 0, xs.size(0)).cpu()[:, 1]
                        AUCval_source.append(
                            utils.roc_auc_score_trainval(
                                ys_cpu.detach().numpy(), y_pre.detach().numpy()
                            )
                        )
                        APRval_source.append(
                            average_precision_score(
                                ys_cpu.detach().numpy(), y_pre.detach().numpy()
                            )
                        )
                        running_loss_valid.append(loss.item())
                        running_dann_valid.append(-loss_t.item())
                        running_c_valid.append(loss_c.item())

            if phase == "train":
                epoch_loss = np.mean(np.array(running_loss_train))
                epoch_dann = np.mean(np.array(running_dann_train))
                epoch_c = np.mean(np.array(running_c_train))
                auc = np.mean(np.array(AUCtrain_source))
                aupr = np.mean(np.array(APRtrain_source))
            else:
                epoch_loss = np.mean(np.array(running_loss_valid))
                epoch_dann = np.mean(np.array(running_dann_valid))
                epoch_c = np.mean(np.array(running_c_valid))
                auc = np.mean(np.array(AUCval_source))
                aupr = np.mean(np.array(APRval_source))

            train_AUCs = [epoch, epoch_loss, epoch_dann, epoch_c, phase, auc, aupr]
            print(
                "Epoch:{} Phase:{} AUC: {:.6f} AUPR: {:.6f} ".format(
                    epoch, phase, auc, aupr
                )
            )
            save_AUCs(train_AUCs, file_AUCs)

            if (phase == "val") and (epoch_loss < best_loss or best_auc < auc):
                best_auc = auc
                best_loss = epoch_loss

    return encoder, predictor, adentropy_p
