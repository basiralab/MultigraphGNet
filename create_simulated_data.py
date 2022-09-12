import numpy as np
import config
import os


if __name__ == "__main__":
    asd_view_means = [
        0.08625711,
        0.00957597,
        0.82772245,
        0.17007891,
        0.21787844,
        0.0200403,
    ]
    asd_view_stds = [
        0.07451019,
        0.00952655,
        0.94726076,
        0.27995504,
        0.19697252,
        0.01868246,
    ]
    nc_view_means = [
        0.08241909,
        0.00913697,
        0.83621246,
        0.16708059,
        0.21835637,
        0.01965899,
    ]
    nc_view_stds = [
        0.06969878,
        0.00847601,
        0.95460409,
        0.27907256,
        0.19909331,
        0.01851708,
    ]

    asd_simulated_data = np.stack(
        [
            np.random.normal(mean, std, (config.N_SAMPLES, config.N_ROI, config.N_ROI))
            for mean, std in zip(asd_view_means, asd_view_stds)
        ],
        axis=3,
    )
    nc_simulated_data = np.stack(
        [
            np.random.normal(mean, std, (config.N_SAMPLES, config.N_ROI, config.N_ROI))
            for mean, std in zip(nc_view_means, nc_view_stds)
        ],
        axis=3,
    )

    if not os.path.isdir("simulated_data"):
        os.makedirs("simulated_data")

    np.save("simulated_data/asd_simulated_data.npy", asd_simulated_data)
    np.save("simulated_data/nc_simulated_data.npy", nc_simulated_data)
