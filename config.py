import warnings
warnings.filterwarnings("ignore")

# Change the paths and names below according to your dataset
class DatasetClass1:
    path = "simulated_data/asd_simulated_data.npy"
    name = "simulated_asd"


class DatasetClass2:
    path = "simulated_data/nc_simulated_data.npy"
    name = "simulated_nc"


# Specify which class to train and test DGN/RDGN on
DGN_RDGN_TRAIN_CLASS = DatasetClass1

N_SAMPLES = 10
N_VIEWS = 6
N_ROI = 35

# Number of folds for cross-validation
N_FOLDS = 5

# Seed for K-Fold CV
# IMPORTANT: Train-test splitting relies on this seed.
# To prevent mixing train-test sets, keep the seed same for DGN/RDGN and classifier training.
SEED = 7

N_EPOCHS = 3

SAVE_WEIGHTS_EVERY_NTH_EPOCH = 10

# Number of samples to augment using our trained RDGN net
K = 15
