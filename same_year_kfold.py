import sys
import numpy as np
import torch

from sklearn.model_selection import KFold

from main import trainTestModel, plotMetrics, checkOutliers

# Basically the same code as in main.py, but using different random seeds.
def main(argv):
    year = int(argv[1]) if len(argv) > 1 else 2018
    model_name = argv[2] if len(argv) > 2 else "MLP"
    rng_seed = int(argv[3]) if len(argv) > 3 else 42
    # If seed is given, plots are not shown    
    show_plots = False if len(argv) > 3 else True

    grid_mean = False  # whether or not to use grid mean corrected VV and VH data
    n_epochs = 100

    check_outliers = False # Show percentage of outliers on false negatives

    print(f'(Random seed set to {rng_seed})')
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)

    dataset = np.load(f'Colza_DB/Colza_data_{year}.npz', allow_pickle=True)
    X_SAR, X_NDVI, y_multi, id_parcel = dataset["X_SAR"], dataset["X_NDVI"], dataset["y"], dataset["id_parcels"]

    dates = dataset["dates_SAR"]

    if not grid_mean:
        X_SAR = X_SAR[:, :, (0, 2)]  # Remove VV-grid_mean and VH-grid_mean

    # Pre-process labels: binarize with "CZH" as positive class
    y = np.zeros(y_multi.shape)
    y[y_multi == "CZH"] = 1

    # Train-test split
    kf = KFold(n_splits=5) #, random_state=rng_seed, shuffle=True
    batch_size = 16

    for i, (idx_train, idx_test) in enumerate(kf.split(X_SAR)):
        print(f"\n---------- Fold {i} ----------")
        X_SAR_train, X_SAR_test = X_SAR[idx_train], X_SAR[idx_test]
        # X_NDVI_train, X_NDVI_test = X_NDVI[idx_train], X_NDVI[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        x_train = torch.Tensor(X_SAR_train)
        y_train = torch.Tensor(y_train).long()
        # y_train = F.one_hot(y_train, num_classes=n_classes).float()

        x_test = torch.Tensor(X_SAR_test)  # transform to torch tensor
        y_test = torch.FloatTensor(y_test).long()
        # y_test = F.one_hot(y_test, num_classes=n_classes).float()

        # Permute channel and time dimensions
        x_train = x_train.permute((0, 2, 1))
        x_test = x_test.permute((0, 2, 1))

        path = "model_weights/"
        filename = model_name+f'_SAR_{year}part_{n_epochs}ep_{x_train.shape[-2]}ch_seed{rng_seed}_fold{i}'
        y_pred = trainTestModel(model_name, path+filename, x_train, x_test,
                                y_train, y_test, dates, dates, n_epochs, batch_size)

        # Metrics
        if show_plots:
            path = "results/1_same_year/"
            plotMetrics(y_test,  y_multi[idx_test], y_pred,path,filename)

        # Check outliers on false negatives
        if check_outliers:
            checkOutliers(y_pred,y_test,idx_test,year)

        # print( model.parameters() )


if __name__ == "__main__":
    main(sys.argv)
