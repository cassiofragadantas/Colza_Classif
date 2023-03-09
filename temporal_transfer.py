import sys
import numpy as np
import torch

from main import trainTestModel, plotMetrics, checkOutliers, removeOutliers


def main(argv):
    if len(argv) > 2:
        year_train, year_test = int(argv[1]), int(argv[2])
    else:
        year_train, year_test = 2018, 2020

    model_name = argv[3] if len(argv) > 3 else "MLP"
    rng_seed = int(argv[4]) if len(argv) > 4 else 42
    show_plots = False if len(argv) > 4 else True
    add_shift = int(argv[5]) if len(argv) > 5 else 0 # forces a time shift of 'add_shift' entries between train and test years
    remove_outliers = True if len(argv) > 6 and argv[6].lower()=='true' else False # Remove outliers from training and test data
    check_outliers = True if len(argv) > 7 and argv[7].lower()=='true' else False # Show percentage of outliers on false negatives
    outlierType = argv[8] if len(argv) > 8 else  'intersect' # Criteria for outlier detection ('VV', 'VH', 'union')


    grid_mean = True # whether or not to use grid mean corrected VV and VH data
    all_orbits = True # whether or not to merge all orbits for a higher temporal resolution
    n_epochs = 100

    print(f'(Random seed set to {rng_seed})')
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)


    # Training data
    dataset = np.load(
        f'Colza_DB/Colza_data_{year_train}.npz', allow_pickle=True)
    X_SAR_train, X_NDVI_train, y_multi_train = dataset["X_SAR"], dataset["X_NDVI"], dataset["y"]
    id_parcels_train, dates_SAR_train, dates_NDVI_train = \
        dataset["id_parcels"], dataset["dates_SAR"], dataset["dates_NDVI"]

    # Test data
    dataset = np.load(
        f'Colza_DB/Colza_data_{year_test}.npz', allow_pickle=True)
    X_SAR_test, X_NDVI_test, y_multi_test = dataset["X_SAR"], dataset["X_NDVI"], dataset["y"]
    id_parcel_test, dates_SAR_test, dates_NDVI_test = \
        dataset["id_parcels"], dataset["dates_SAR"], dataset["dates_NDVI"]

    if remove_outliers:
        X_SAR_train = removeOutliers(X_SAR_train,year_train,outType=outlierType)
        y_multi_train = removeOutliers(y_multi_train,year_train,outType=outlierType)

    # Pre_process data: resize
    # 2018: SAR (80288, 152, 4), NDVI (80288, 78)
    # 2019: SAR (72506, 135, 4), NDVI (72506, 80)
    # 2020: SAR (97971, 150, 4), NDVI (97971, 71)
    if model_name != 'LTAE':
        # Downsize to smaller year between train and test
        # diff = X_SAR_train.shape[1] - X_SAR_test.shape[1]
        # if diff > 0:
        #     X_SAR_train = X_SAR_train[:, (diff//2):-(diff-diff//2), :]
        # elif diff < 0:
        #     X_SAR_test = X_SAR_test[:, -(diff//2):(diff-diff//2), :]

        if add_shift!=0:
            if (year_train != 2018) or (year_test != 2020):
                print('Temporal shift experiment is only available for training at 2018 and test at 2020')
                return
            if add_shift != 4 and add_shift != 8 and add_shift != 15:
                print('Shift has to be of 8 or 15 samples. Reseting it to 15.')
                add_shift = 15
            if add_shift == 4: # about 7-days shift
                X_SAR_train = X_SAR_train[:, 12:-5, :]
                X_SAR_test = X_SAR_test[:, 15:, :]                
            elif add_shift == 8: # about 15-days shift
                X_SAR_train = X_SAR_train[:, 8:-9, :]
                X_SAR_test = X_SAR_test[:, 15:, :]
            else: # about 30-days shift
                X_SAR_train = X_SAR_train[:, 2:-15, :]
                X_SAR_test = X_SAR_test[:, 15:, :]
        else:
            # Downsize always to 135 (i.e, smallest year overall) for full compatibily
            diff = X_SAR_train.shape[1] - 135
            if diff > 0:
                X_SAR_train = X_SAR_train[:, (diff//2):-(diff-diff//2), :]
            diff = X_SAR_test.shape[1] - 135
            if diff > 0:
                X_SAR_test = X_SAR_test[:, (diff//2):-(diff-diff//2), :]


    if not all_orbits:
        X_SAR_train = X_SAR_train[:,::3,:]
        X_SAR_test = X_SAR_test[:,::3,:]


    if not grid_mean:
        X_SAR_train = X_SAR_train[:, :, (0, 2)]  # Remove VV-grid_mean and VH-grid_meanx
    # Pre_process data: rescale
    x_train = X_SAR_train/np.percentile(X_SAR_train, 99)
    x_train[x_train > 1] = 1
    x_train = torch.Tensor(x_train)
    # Pre-process labels: binarize with "CZH" as positive class
    y_train = np.zeros(y_multi_train.shape)
    y_train[y_multi_train == "CZH"] = 1
    y_train = torch.Tensor(y_train).long()

    # Permute channel and time dimensions
    x_train = x_train.permute((0,2,1))


    # Pre_process data: rescaling
    if not grid_mean:
        X_SAR_test = X_SAR_test[:, :, (0, 2)]  # Remove VV-grid_mean and VH-grid_meanx    
    x_test = X_SAR_test/np.percentile(X_SAR_test, 99)
    x_test[x_test > 1] = 1
    x_test = torch.Tensor(x_test)  # transform to torch tensor
    # Pre-process labels: binarize with "CZH" as positive class
    y_test = np.zeros(y_multi_test.shape)
    y_test[y_multi_test == "CZH"] = 1
    y_test = torch.Tensor(y_test).long()
    y_test = torch.Tensor(y_test).long()

    # Permute channel and time dimensions
    x_test = x_test.permute((0,2,1))

    # Train and test model
    path = "model_weights/"
    filename = model_name + f'_SAR_{year_train}_{n_epochs}ep_{x_train.shape[-2]}ch_seed{rng_seed}'
    if not all_orbits:
        filename = filename + '_singleOrb'
    if add_shift != 0:
        filename = filename + f'_shift{add_shift}'
    if remove_outliers:
        filename = filename + f'_noOutliers_{outlierType}'
    print(filename)
    y_pred = trainTestModel(model_name,path+filename,x_train,x_test,y_train,y_test,dates_SAR_train,dates_SAR_test,n_epochs)

    # Metrics
    if show_plots:
        path = "results/2_temporal_transfer/"
        plotMetrics(y_test, y_multi_test, y_pred, path, filename)

    # Check outliers on false negatives
    if check_outliers:
        checkOutliers(y_pred,y_test,np.arange(len(y_pred)),year_test,outlierType)

    # print( model.parameters() )
    # exit()


if __name__ == "__main__":
    main(sys.argv)
