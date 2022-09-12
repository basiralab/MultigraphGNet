import logging
import matplotlib.pyplot as plt
import torch
import os
import numpy as np


def get_logger() -> logging.Logger:
    path = "train_logs.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s\n%(message)s")

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def save_weights(fold_num, dgn, rdgn, name, seed, classname):
    torch.save(
        rdgn.state_dict(),
        os.path.join(
            f"fold_{fold_num}_seed_{seed}_classname_{classname}",
            f"rdgn_{name}_fold_{fold_num}_seed_{seed}_classname_{classname}.pt",
        ),
    )
    torch.save(
        dgn.state_dict(),
        os.path.join(
            f"fold_{fold_num}_seed_{seed}_classname_{classname}",
            f"dgn_{name}_fold_{fold_num}_seed_{seed}_classname_{classname}.pt",
        ),
    )

    print(f"Weights saved with name {name}\n")


def plot(random_test_sample, cbt, data_hat, mae, fold_num, seed, classname):
    figure = plt.figure(figsize=(10, 10))
    cols, rows = data_hat.shape[2], 3

    for i in range(1, cols + 1):
        figure.add_subplot(rows, cols, i)
        plt.xlabel(f"View {i}", fontsize=8)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(random_test_sample[:, :, i - 1].cpu().detach().numpy())

    for i in range(1, cols + 1):
        figure.add_subplot(rows, cols, i + cols)
        plt.xlabel(
            f"Reconstructed View {i}\nMAE: {mae[i - 1].item():.2f}",
            fontsize=8,
        )
        plt.xticks([])
        plt.yticks([])

        plt.imshow(data_hat[:, :, i - 1].cpu().detach().numpy())

    figure.add_subplot(rows, cols, 2 * cols + 1)
    plt.xlabel("CBT")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cbt.cpu().detach().numpy())

    plt.savefig(
        os.path.join(
            f"fold_{fold_num}_seed_{seed}_classname_{classname}",
            f"random_test_sample_results_fold_{fold_num}_seed_{seed}_classname_{classname}.png",
        )
    )


def generate_cbt_median(model, train_data, device):
    """
    Generate optimized CBT for the training set (use post training refinement)
    Args:
        model: trained DGN model
        train_data: list of data objects

    Taken from https://github.com/basiralab/DGN
    """
    model.eval()
    cbts = []
    train_data = [d.to(device) for d in train_data]
    for data in train_data:
        cbt = model(data)
        cbts.append(np.array(cbt.cpu().detach()))
    final_cbt = torch.tensor(np.median(cbts, axis=0), dtype=torch.float32).to(device)

    return final_cbt


def mean_frobenious_distance(generated_cbt, test_data):
    """
    Calculate the mean Frobenious distance between the CBT and test subjects (all views)
    Args:
        generated_cbt: trained DGN model
        test_data: list of data objects

    Taken from https://github.com/basiralab/DGN
    """
    frobenius_all = []
    for data in test_data:
        views = data.con_mat
        for index in range(views.shape[2]):
            diff = torch.abs(views[:, :, index] - generated_cbt)
            diff = diff * diff
            sum_of_all = diff.sum()
            d = torch.sqrt(sum_of_all)
            frobenius_all.append(d)
    return sum(frobenius_all) / len(frobenius_all)


def vectorize(m):
    new_m = m.copy()
    return new_m[np.triu_indices(new_m.shape[0], k=1)].flatten()


def reconstruct(model, cbt):
    cbt_c = cbt.copy()
    cbt_c = torch.from_numpy(cbt_c).float()
    cbt_c = cbt_c.unsqueeze(0).unsqueeze(0)
    data_hat = model(cbt_c)
    return data_hat.squeeze().permute(1, 2, 0).detach().cpu().numpy()


def cosine_similarity_matrix(dataset):
    m = torch.zeros((dataset.shape[0], dataset.shape[0]), dtype=torch.float)
    for row in range(dataset.shape[0]):
        for col in range(dataset.shape[0]):
            m[row, col] = torch.cosine_similarity(
                dataset[row].flatten(), dataset[col].flatten(), dim=0
            )

    return m.cpu().detach().numpy()


def plot_cosine_similarity_matrix(cs_m):
    plt.figure()
    plt.imshow(cs_m)
    plt.colorbar()
    plt.savefig("cs_m.png")
