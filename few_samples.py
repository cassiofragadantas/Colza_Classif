import sys
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from main import trainTestModel, plotMetrics


def main(argv):
    year = int(argv[1]) if len(argv) > 1 else 2018
    model_name = argv[2] if len(argv) > 2 else "MLP"
    rng_seed = int(argv[3]) if len(argv) > 3 else 42
    # If seed is given, plots are not shown
    show_plots = False if len(argv) > 3 else True

    train_sizes = [100, 300, 500, 1000]

    grid_mean = True  # whether or not to use grid mean corrected VV and VH data
    renormalize = True  # Normalize data to [0, 1] ignoring outliers (1%)
    balanced = False  # train-test sets balanced for class distribution
    n_epochs = 100
    batch_size = 20  # Preferably a divisor of all train_sizes

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

    for train_size in train_sizes:
        print(f'\n============ {train_size} training samples ============')
        # Train-test split
        indices = np.arange(y.shape[0])
        train_size = int(batch_size * round(train_size / batch_size))
        if balanced:
            idx_train = np.concatenate((
                np.random.choice(np.where(y == 1)[
                                 0], train_size//2, replace=False),
                np.random.choice(np.where(y == 0)[0], (train_size+1)//2, replace=False))
            )
            idx_test = np.delete(indices, idx_train)
            X_SAR_train, X_SAR_test = X_SAR[idx_train], X_SAR[idx_test]
            X_NDVI_train, X_NDVI_test = X_NDVI[idx_train], X_NDVI[idx_test]
            y_train, y_test = y[idx_train], y[idx_test]
        else:
            X_SAR_train, X_SAR_test, X_NDVI_train, X_NDVI_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X_SAR, X_NDVI, y, indices, train_size=train_size, random_state=rng_seed)

        # Pre_process data
        if model_name != 'LTAE':
            # Downsize always to 135 (i.e, smallest year overall) for full compatibily
            diff = X_SAR_train.shape[1] - 135
            if diff > 0:
                X_SAR_train = X_SAR_train[:, (diff//2):-(diff-diff//2), :]
            diff = X_SAR_test.shape[1] - 135
            if diff > 0:
                X_SAR_test = X_SAR_test[:, (diff//2):-(diff-diff//2), :]

        if renormalize:
            # Pre_process data: rescale
            x_train = X_SAR_train/np.percentile(X_SAR_train, 99)
            x_train[x_train > 1] = 1
            # Pre-process labels: binarize with "CZH" as positive class
            y_train = np.zeros(y_multi[idx_train].shape)
            y_train[y_multi[idx_train] == "CZH"] = 1

            # Pre_process data: rescaling
            x_test = X_SAR_test/np.percentile(X_SAR_test, 99)
            x_test[x_test > 1] = 1
            # Pre-process labels: binarize with "CZH" as positive class
            y_test = np.zeros(y_multi[idx_test].shape)
            y_test[y_multi[idx_test] == "CZH"] = 1

        x_train = torch.Tensor(X_SAR_train)
        y_train = torch.Tensor(y_train).long()
        # y_train = F.one_hot(y_train, num_classes=n_classes).float()

        x_test = torch.Tensor(X_SAR_test)  # transform to torch tensor
        y_test = torch.FloatTensor(y_test).long()
        # y_test = F.one_hot(y_test, num_classes=n_classes).float()

        # Permute channel and time dimensions
        x_train = x_train.permute((0, 2, 1))
        x_test = x_test.permute((0, 2, 1))

        # Train and test model
        distrib = 'eq' if balanced else 'nonEq'
        path = "model_weights/"
        filename =  model_name + \
            f'_SAR_{year}-{train_size}sp-{distrib}_{n_epochs}ep_{x_train.shape[-2]}ch_seed{rng_seed}'
        y_pred = trainTestModel(model_name, path+filename, x_train, x_test,
                                y_train, y_test, dates, dates, n_epochs, batch_size)

        # Metrics
        if show_plots:
            path = "results/3_few_samples/"
            plotMetrics(y_test,  y_multi[idx_test], y_pred, path, filename)

    # print( model.parameters() )


if __name__ == "__main__":
    main(sys.argv)
