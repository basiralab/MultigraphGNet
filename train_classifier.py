"""
    Function to evaluate the effectiveness of our MultigraphGNet framework.
    ---------------------------------------------------------------------
    We train two independent SVM classifiers using
        1) one global CBT from each class (one-shot CBT baseline)
        2) samples augmented by our trained RDGN net.
            We augment k samples, you can specify the number of augmented samples by changing the config.K
    ---------------------------------------------------------------------
    Copyright 2022 Furkan Pala, Istanbul Technical University.
    All rights reserved.
"""


from utils import generate_cbt_median, vectorize, reconstruct
import os
import numpy as np
import torch
from data import read_simulated_dataset, cast_data
from sklearn.model_selection import KFold
from model import DGN, UNet
import config
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import json


def train_classifier(fold_num, seed, k):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    print("Seed", seed)
    print("Fold", fold_num)
    print("k", k)

    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

    nc_data = read_simulated_dataset(config.DatasetClass1.path)
    asd_data = read_simulated_dataset(config.DatasetClass2.path)

    n_samples_nc, n_roi_nc, _, n_views_nc = nc_data.shape
    n_samples_asd, n_roi_asd, _, n_views_asd = asd_data.shape

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

    nc_folds = list(kfold.split(nc_data))
    asd_folds = list(kfold.split(asd_data))

    nc_train_ind, nc_test_ind = nc_folds[fold_num]
    asd_train_ind, asd_test_ind = asd_folds[fold_num]

    nc_train, nc_test = nc_data[nc_train_ind], nc_data[nc_test_ind]
    asd_train, asd_test = asd_data[asd_train_ind], asd_data[asd_test_ind]

    nc_train, nc_test = torch.from_numpy(nc_train), torch.from_numpy(nc_test)
    asd_train, asd_test = torch.from_numpy(asd_train), torch.from_numpy(asd_test)

    nc_train_views_min, nc_train_views_max = nc_train.amin(
        dim=(0, 1, 2)
    ), nc_train.amax(dim=(0, 1, 2))

    asd_train_views_min, asd_train_views_max = asd_train.amin(
        dim=(0, 1, 2)
    ), asd_train.amax(dim=(0, 1, 2))

    nc_train = (nc_train - nc_train_views_min) / (
        nc_train_views_max - nc_train_views_min
    )
    nc_test = (nc_test - nc_train_views_min) / (nc_train_views_max - nc_train_views_min)

    asd_train = (asd_train - asd_train_views_min) / (
        asd_train_views_max - asd_train_views_min
    )
    asd_test = (asd_test - asd_train_views_min) / (
        asd_train_views_max - asd_train_views_min
    )

    nc_train_casted, nc_test_casted = cast_data(nc_train, device), cast_data(
        nc_test, device
    )
    asd_train_casted, asd_test_casted = cast_data(asd_train, device), cast_data(
        asd_test, device
    )

    nc_dgn_weights_path = os.path.join(
        f"fold_{fold_num}_seed_{seed}_classname_{config.DatasetClass1.name}",
        f"dgn_best_mae_fold_{fold_num}_seed_{seed}_classname_{config.DatasetClass1.name}.pt",
    )

    nc_rdgn_weights_path = os.path.join(
        f"fold_{fold_num}_seed_{seed}_classname_{config.DatasetClass1.name}",
        f"rdgn_best_mae_fold_{fold_num}_seed_{seed}_classname_{config.DatasetClass1.name}.pt",
    )
    asd_dgn_weights_path = os.path.join(
        f"fold_{fold_num}_seed_{seed}_classname_{config.DatasetClass2.name}",
        f"dgn_best_mae_fold_{fold_num}_seed_{seed}_classname_{config.DatasetClass2.name}.pt",
    )
    asd_rdgn_weights_path = os.path.join(
        f"fold_{fold_num}_seed_{seed}_classname_{config.DatasetClass2.name}",
        f"rdgn_best_mae_fold_{fold_num}_seed_{seed}_classname_{config.DatasetClass2.name}.pt",
    )

    nc_dgn = DGN(n_views_nc, 36, 24, 5).to(device)
    nc_rdgn = UNet(1, n_views_nc).to(device)

    asd_dgn = DGN(n_views_asd, 36, 24, 5).to(device)
    asd_rdgn = UNet(1, n_views_asd).to(device)

    nc_dgn.eval()
    nc_rdgn.eval()
    asd_dgn.eval()
    asd_rdgn.eval()

    nc_dgn.load_state_dict(
        torch.load(nc_dgn_weights_path, map_location=torch.device("cpu"))
    )
    nc_rdgn.load_state_dict(
        torch.load(nc_rdgn_weights_path, map_location=torch.device("cpu"))
    )
    asd_dgn.load_state_dict(
        torch.load(asd_dgn_weights_path, map_location=torch.device("cpu"))
    )
    asd_rdgn.load_state_dict(
        torch.load(asd_rdgn_weights_path, map_location=torch.device("cpu"))
    )

    nc_cbt_train = (
        generate_cbt_median(nc_dgn, nc_train_casted, device).cpu().detach().numpy()
    )
    asd_cbt_train = (
        generate_cbt_median(asd_dgn, asd_train_casted, device).cpu().detach().numpy()
    )

    nc_train_feats = vectorize(nc_cbt_train)
    asd_train_feats = vectorize(asd_cbt_train)

    nc = 0
    asd = 1

    svc = SVC()
    svc.fit([nc_train_feats, asd_train_feats], [nc, asd])

    nc_test_cbts = np.array(
        [
            nc_dgn(nc_test_sample).cpu().detach().numpy()
            for nc_test_sample in nc_test_casted
        ]
    )
    asd_test_cbts = np.array(
        [
            asd_dgn(asd_test_sample).cpu().detach().numpy()
            for asd_test_sample in asd_test_casted
        ]
    )

    nc_test_feats = np.array([vectorize(nc_test_cbt) for nc_test_cbt in nc_test_cbts])
    asd_test_feats = np.array(
        [vectorize(asd_test_cbt) for asd_test_cbt in asd_test_cbts]
    )

    test_feats = np.concatenate([nc_test_feats, asd_test_feats], axis=0)
    test_labels = np.concatenate(
        [np.full(nc_test_feats.shape[0], nc), np.full(asd_test_feats.shape[0], asd)]
    )

    preds = svc.predict(test_feats)

    acc = accuracy_score(test_labels, preds)
    prec = precision_score(test_labels, preds)
    rec = recall_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)

    print("CBT Oneshot results")
    print(
        f"{np.count_nonzero((preds - test_labels) == 0)} / {test_labels.shape[0]} samples correctly classified"
    )
    print(f"Acc: {acc}")
    print(f"Prec: {prec}")
    print(f"Rec: {rec}")
    print(f"F1: {f1}")

    nc_cbt_mean, nc_cbt_std = np.mean(vectorize(nc_cbt_train)), np.std(
        vectorize(nc_cbt_train)
    )
    asd_cbt_mean, asd_cbt_std = np.mean(vectorize(asd_cbt_train)), np.std(
        vectorize(asd_cbt_train)
    )
    nc_train_noised_cbts = []
    asd_train_noised_cbts = []

    for ith_aug in range(k):
        nc_noise = (
            np.random.normal(nc_cbt_mean, nc_cbt_std, n_roi_nc * (n_roi_nc - 1) // 2)
            * 0.2
        )
        asd_noise = (
            np.random.normal(
                asd_cbt_mean, asd_cbt_std, n_roi_asd * (n_roi_asd - 1) // 2
            )
            * 0.2
        )

        nc_noise[nc_noise < 0] = 0
        asd_noise[asd_noise < 0] = 0

        nc_noise_m = np.zeros(nc_cbt_train.shape)
        asd_noise_m = np.zeros(asd_cbt_train.shape)

        nc_noise_m[np.triu_indices_from(nc_noise_m, k=1)] = nc_noise
        nc_noise_m = nc_noise_m + nc_noise_m.T

        asd_noise_m[np.triu_indices_from(asd_noise_m, k=1)] = asd_noise
        asd_noise_m = asd_noise_m + asd_noise_m.T

        nc_cbt_e = nc_cbt_train + nc_noise_m
        asd_cbt_e = asd_cbt_train + asd_noise_m

        nc_train_noised_cbts.append(nc_cbt_e)
        asd_train_noised_cbts.append(asd_cbt_e)

    nc_train_aug = np.stack(
        [reconstruct(nc_rdgn, cbt) for cbt in nc_train_noised_cbts], axis=0
    )
    asd_train_aug = np.stack(
        [reconstruct(asd_rdgn, cbt) for cbt in asd_train_noised_cbts],
        axis=0,
    )

    nc_train_aug_feats = np.array([vectorize(sample) for sample in nc_train_aug])
    asd_train_aug_feats = np.array([vectorize(sample) for sample in asd_train_aug])

    nc_test_aug_feats = np.array(
        [vectorize(sample.cpu().detach().numpy()) for sample in nc_test]
    )
    asd_test_aug_feats = np.array(
        [vectorize(sample.cpu().detach().numpy()) for sample in asd_test]
    )

    svc_aug = SVC()
    svc_aug_train_feats = np.concatenate([nc_train_aug_feats, asd_train_aug_feats])
    svc_aug_train_labels = np.concatenate(
        [
            np.full(nc_train_aug_feats.shape[0], nc),
            np.full(asd_train_aug_feats.shape[0], asd),
        ]
    )

    svc_aug.fit(svc_aug_train_feats, svc_aug_train_labels)

    svc_aug_test_feats = np.concatenate([nc_test_aug_feats, asd_test_aug_feats])
    svc_aug_test_labels = np.concatenate(
        [
            np.full(nc_test_aug_feats.shape[0], nc),
            np.full(asd_test_aug_feats.shape[0], asd),
        ]
    )

    preds_aug = svc_aug.predict(svc_aug_test_feats)

    acc_aug = accuracy_score(svc_aug_test_labels, preds_aug)
    prec_aug = precision_score(svc_aug_test_labels, preds_aug)
    rec_aug = recall_score(svc_aug_test_labels, preds_aug)
    f1_aug = f1_score(svc_aug_test_labels, preds_aug)

    print("Augmented results")
    print(
        f"{np.count_nonzero((preds_aug - svc_aug_test_labels) == 0)} / {svc_aug_test_labels.shape[0]} samples correctly classified"
    )
    print(f"Acc: {acc_aug}")
    print(f"Prec: {prec_aug}")
    print(f"Rec: {rec_aug}")
    print(f"F1: {f1_aug}")

    return (
        acc,
        prec,
        rec,
        f1,
        acc_aug,
        prec_aug,
        rec_aug,
        f1_aug,
        preds,
        test_labels,
        preds_aug,
        svc_aug_test_labels,
    )


if __name__ == "__main__":
    k = config.K
    seed = config.SEED

    results = {
        "seed": seed,
        "k": k,
        "baseline_acc": [],
        "baseline_prec": [],
        "baseline_rec": [],
        "baseline_f1": [],
        "aug_acc": [],
        "aug_prec": [],
        "aug_rec": [],
        "aug_f1": [],
    }

    for i in range(config.N_FOLDS):
        (
            acc,
            prec,
            rec,
            f1,
            acc_aug,
            prec_aug,
            rec_aug,
            f1_aug,
            preds,
            test_labels,
            preds_aug,
            aug_test_labels,
        ) = train_classifier(i, seed, k)
        print()

        results["baseline_acc"].append(acc)
        results["baseline_prec"].append(prec)
        results["baseline_rec"].append(rec)
        results["baseline_f1"].append(f1)

        results["aug_acc"].append(acc_aug)
        results["aug_prec"].append(prec_aug)
        results["aug_rec"].append(rec_aug)
        results["aug_f1"].append(f1_aug)

    results["baseline_acc"].append(np.mean(results["baseline_acc"]))
    results["baseline_prec"].append(np.mean(results["baseline_prec"]))
    results["baseline_rec"].append(np.mean(results["baseline_rec"]))
    results["baseline_f1"].append(np.mean(results["baseline_f1"]))

    results["aug_acc"].append(np.mean(results["aug_acc"]))
    results["aug_prec"].append(np.mean(results["aug_prec"]))
    results["aug_rec"].append(np.mean(results["aug_rec"]))
    results["aug_f1"].append(np.mean(results["aug_f1"]))

    if not os.path.isdir("classifier_results"):
        os.makedirs("classifier_results")

    with open(
        f"classifier_results/classifier_results_seed_{seed}_k_{k}.json", "w"
    ) as f:
        f.write(json.dumps(results))
