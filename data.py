import os
import scipy
from torch_geometric.data import Data
import numpy as np
from typing import List
import torch


def read_all_dataset(root, read_indices=None, connection_mask=None) -> List[np.array]:
    """
    For reading the dataset that we have used in the paper
    """
    files = os.listdir(root)
    all_data = []
    files = sorted(files, key=lambda f: int(f.split(".mat")[0].split("Sub")[1]))[:155]
    for i, file in enumerate(files):
        if read_indices == None:
            mvbn = scipy.io.loadmat(root + "/" + file)["views"]
            if connection_mask is not None:
                mvbn[connection_mask != 1] = 0
            all_data.append(mvbn)
        else:
            if i in read_indices:
                mvbn = scipy.io.loadmat(root + "/" + file)["views"]
                if connection_mask is not None:
                    mvbn[connection_mask != 1] = 0
                all_data.append(mvbn)

    return [np.array(data) for data in all_data]


def read_simulated_dataset(path) -> np.array:
    return np.load(path)


def cast_data(
    array_of_tensors, device, subject_type=None, flat_mask=None
) -> List[Data]:

    """
    Create data objects  for the DGN
    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
    """
    N_ROI = array_of_tensors[0].shape[0]
    CHANNELS = array_of_tensors[0].shape[2]
    dataset = []
    for mat in array_of_tensors:
        edge_index = torch.zeros((2, N_ROI * N_ROI), dtype=torch.long)
        edge_attr = torch.zeros((N_ROI * N_ROI, CHANNELS), dtype=torch.float)
        x = torch.zeros((N_ROI, 1), dtype=torch.float)
        row_ind, col_ind = torch.triu_indices(N_ROI, N_ROI, offset=1)
        triu = mat[row_ind, col_ind]

        counter = 0
        for i in range(N_ROI):
            for j in range(N_ROI):
                edge_index[:, counter] = torch.tensor([i, j])
                edge_attr[counter, :] = mat[i, j]
                counter += 1

        # Fill node feature matrix (no features every node is 1)
        for i in range(N_ROI):
            x[i, 0] = 1

        if flat_mask is not None:
            edge_index_masked = []
            edge_attr_masked = []
            for i, val in enumerate(flat_mask):
                if val == 1:
                    edge_index_masked.append(edge_index[:, i])
                    edge_attr_masked.append(edge_attr[i, :])
            edge_index = np.array(edge_index_masked).T
            edge_attr = edge_attr_masked

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            con_mat=mat,
            label=subject_type,
            triu=triu,
        )
        dataset.append(data.to(device))

    return dataset
