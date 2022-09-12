"""
    Main function of the MultigraphGNet framework for 
    Predicting Brain Multigraph Population From a Single Graph Template for Boosting One-Shot Classification.
    ---------------------------------------------------------------------

    This file contains the driver code for the training and testing process of our framework.

    Data:
        Dataset is represented as a 4 dimensional tensor of shape (N_SAMPLES, N_ROI, N_ROI, N_VIEWS).
        To represent each sample as a graph, we convert to torch_geometric.data.Data format using utils.cast_data function.
      
        To cerate a simulated dataset, use create_simulated_data.py script.

        To evaluate the effectiveness of our framework, we trained an independent classifier (SVM)
        to distinguish between ASD and NC subjects using one global CBT and samples augmented by our framework.
        For more details, check the paper.
        To simulate this classification task, create_simulated_data.py script generates sets with two classes: class1 and class2.
        Thus, you need to train our framework on two classes seperately.
            While doing so, do not forget to change the DGN_RDGN_TRAIN_CLASS variable in config.py

        IMPORTANT:
            When training our framework with a simulated data, results may not be interpretable 
                since we sample the simulated data from a normal distribution.

    Model:
        Our framework consists of two neural networks: DGN and RDGN.
        DGN is responsible for creating a subject specific CBT of shape (N_ROI, N_ROI)
            which is a single-view brain graph of the multi-view subject of shape (N_ROI, N_ROI, N_VIEWS)
        DGN is a graph neual network (GNN).

        RDGN is responsible for reverse mapping the CBT created by DGN to the original views.
        RDGN has the architecture of U-Net.
    ---------------------------------------------------------------------
    Copyright 2022 Furkan Pala, Istanbul Technical University.
    All rights reserved.
"""

from utils import save_weights, plot
import random
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from data import read_simulated_dataset, cast_data
from sklearn.model_selection import KFold
from model import DGN, UNet
import config


def main():
    seed = config.SEED

    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    dataset_path = config.DGN_RDGN_TRAIN_CLASS.path
    class_name = config.DGN_RDGN_TRAIN_CLASS.name

    # NOTE: if you are using a custom/real dataset,
    # change reading function
    data = read_simulated_dataset(dataset_path)

    n_samples, n_roi, _, n_views = data.shape

    fold_maes, fold_stds = [], []

    kfold = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=seed)
    for fold_num, (train_index, test_index) in enumerate(kfold.split(data)):
        os.makedirs(
            f"fold_{fold_num}_seed_{seed}_classname_{class_name}", exist_ok=True
        )

        train_data, test_data = data[train_index], data[test_index]

        train_data = torch.from_numpy(train_data).float()
        test_data = torch.from_numpy(test_data).float()

        train_views_min, train_views_max = train_data.amin(
            dim=(0, 1, 2)
        ), train_data.amax(dim=(0, 1, 2))

        train_data = (train_data - train_views_min) / (
            (train_views_max - train_views_min)
        )
        test_data = (test_data - train_views_min) / (
            (train_views_max - train_views_min)
        )

        train_mean = train_data.mean(dim=(0, 1, 2))

        train_casted, test_casted = (
            cast_data(train_data, device),
            cast_data(test_data, device),
        )

        loss_weights = torch.tensor(
            np.array(
                list((1 / train_mean) / torch.max(1 / train_mean)) * len(train_data)
            ),
            dtype=torch.float32,
        ).to(device)

        targets = [
            torch.tensor(tensor, dtype=torch.float32).to(device)
            for tensor in train_data
        ]
        targets = torch.cat(targets, axis=2).permute((2, 1, 0))

        dgn = DGN(n_views, 36, 24, 5).to(device)
        rdgn = UNet(1, n_views).to(device)

        optimizer = torch.optim.AdamW(list(dgn.parameters()) + list(rdgn.parameters()))

        (train_dgn_loss_list, train_mae_list, train_loss_list, test_mae_list) = (
            [],
            [],
            [],
            [],
        )
        best_test_mae = float("inf")
        patience = 0
        for epoch in range(1, config.N_EPOCHS + 1):
            dgn.train()
            rdgn.train()

            avg_train_loss, avg_train_mae, avg_train_dgn_loss, avg_test_mae = (
                0.0,
                0.0,
                0.0,
                0.0,
            )
            for train_sample in train_casted:
                cbt = dgn(train_sample)  # N_ROI, N_ROI
                expanded_cbt = cbt.expand((targets.shape[0], n_roi, n_roi))
                diff = torch.abs(expanded_cbt - targets)  # Absolute difference
                sum_of_all = torch.mul(diff, diff).sum(axis=(1, 2))  # Sum of squares
                l = torch.sqrt(sum_of_all)  # Square root of the sum
                dgn_loss = (l * loss_weights).mean()
                avg_train_dgn_loss += dgn_loss

                cbt = cbt.unsqueeze(0).unsqueeze(0)
                data_hat = rdgn(cbt)  # 1, n_views, N_ROI, N_ROI
                data_hat = data_hat.squeeze().permute(1, 2, 0)
                mae = torch.abs(data_hat - train_sample.con_mat).mean()
                avg_train_mae += mae

                avg_train_loss += dgn_loss + mae

            optimizer.zero_grad()
            avg_train_loss /= len(train_casted)
            avg_train_mae /= len(train_casted)
            avg_train_dgn_loss /= len(train_casted)
            avg_train_loss.backward()
            optimizer.step()

            train_dgn_loss_list.append(avg_train_dgn_loss.item())
            train_mae_list.append(avg_train_mae.item())
            train_loss_list.append(avg_train_loss.item())

            dgn.eval()
            rdgn.eval()
            for test_sample in test_casted:
                cbt = dgn(test_sample)  # N_ROI, N_ROI
                cbt = cbt.unsqueeze(0).unsqueeze(0)
                data_hat = rdgn(cbt)  # 1, n_views, N_ROI, N_ROI
                data_hat = data_hat.squeeze().permute(1, 2, 0)
                mae = torch.abs(data_hat - test_sample.con_mat).mean()
                avg_test_mae += mae

            avg_test_mae /= len(test_casted)
            test_mae_list.append(avg_test_mae.item())

            print(
                f"Fold: {fold_num}\n"
                f"Epoch: {epoch}\n"
                f"Train DGN Loss: {avg_train_dgn_loss}\n"
                f"Train MAE: {avg_train_mae}\n"
                f"Train Loss: {avg_train_loss}\n"
                f"Test MAE: {avg_test_mae}\n"
            )

            with open(
                os.path.join(
                    f"fold_{fold_num}_seed_{seed}_classname_{class_name}",
                    f"train_stats_fold_{fold_num}_seed_{seed}_classname_{class_name}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(
                    {
                        "train_dgn_loss_list": train_dgn_loss_list,
                        "train_mae_list": train_mae_list,
                        "train_loss_list": train_loss_list,
                        "test_mae_list": test_mae_list,
                    },
                    f,
                )

            if avg_test_mae < best_test_mae:
                patience = 0
                best_test_mae = avg_test_mae
                save_weights(fold_num, dgn, rdgn, "best_mae", seed, class_name)
            else:
                patience += 1
                if patience > 10:
                    print("Early Stopping")
                    break

            if epoch % config.SAVE_WEIGHTS_EVERY_NTH_EPOCH == 0:
                save_weights(fold_num, dgn, rdgn, f"epoch_{epoch}", seed, class_name)

        print("TESTING STARTED")

        dgn.eval()
        rdgn.eval()

        dgn.load_state_dict(
            torch.load(
                os.path.join(
                    f"fold_{fold_num}_seed_{seed}_classname_{class_name}",
                    f"dgn_best_mae_fold_{fold_num}_seed_{seed}_classname_{class_name}.pt",
                ),
            )
        )
        rdgn.load_state_dict(
            torch.load(
                os.path.join(
                    f"fold_{fold_num}_seed_{seed}_classname_{class_name}",
                    f"rdgn_best_mae_fold_{fold_num}_seed_{seed}_classname_{class_name}.pt",
                ),
            )
        )

        test_mae_list = []
        for test_sample in test_casted:
            cbt = dgn(test_sample)  # N_ROI,N_ROI
            cbt = cbt.unsqueeze(0).unsqueeze(0)
            data_hat = rdgn(cbt)  # 1,n_views,N_ROI,N_ROI
            data_hat = data_hat.squeeze().permute(1, 2, 0)
            mae = torch.abs(data_hat - test_sample.con_mat).mean()
            test_mae_list.append(mae.item())

        avg_test_mae = np.mean(test_mae_list).item()
        std_test_mae = np.std(test_mae_list).item()

        print(f"Test MAE: {avg_test_mae}\n" f"Test STD: {std_test_mae}\n")

        fold_maes.append(avg_test_mae)
        fold_stds.append(std_test_mae)

        # pick a random test sample
        random_test_sample = random.choice(test_casted)
        cbt = dgn(random_test_sample)  # N_ROI,N_ROI
        cbt = cbt.unsqueeze(0).unsqueeze(0)
        data_hat = rdgn(cbt)  # 1,n_views,N_ROI,N_ROI
        data_hat = data_hat.squeeze().permute(1, 2, 0)
        mae = torch.abs(data_hat - random_test_sample.con_mat).mean(dim=(0, 1))
        plot(
            random_test_sample.con_mat,
            cbt.squeeze(),
            data_hat,
            mae,
            fold_num,
            seed,
            class_name,
        )

    mean_across_folds = np.mean(fold_maes)
    std_across_folds = np.std(fold_maes)

    print(
        f"Avg of MAE across folds: {mean_across_folds}\n"
        f"Std of MAE across folds: {std_across_folds}\n"
    )

    fold_maes.append(mean_across_folds.item())
    fold_stds.append(std_across_folds.item())

    fig, ax = plt.subplots()
    ax.bar(
        np.arange(config.N_FOLDS + 1),
        fold_maes,
        yerr=fold_stds,
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
    )
    ax.set_ylabel("MAE")
    ax.set_xticks(np.arange(config.N_FOLDS + 1))
    ax.set_xticklabels([f"Fold {i}" for i in range(0, config.N_FOLDS)] + ["Avg. folds"])
    ax.set_title("Average MAEs and stds")
    ax.yaxis.grid(True)

    if not os.path.isdir("dgn_rdgn_results"):
        os.makedirs("dgn_rdgn_results")

    plt.savefig(
        os.path.join(
            "dgn_rdgn_results",
            f"bar_plot_seed_{seed}_classname_{class_name}.png",
        )
    )


if __name__ == "__main__":
    main()
