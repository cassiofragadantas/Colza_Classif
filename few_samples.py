import sys
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from main import trainTestModel, plotMetrics, checkOutliers, removeOutliers, countOutliers


def main(argv):
    year = int(argv[1]) if len(argv) > 1 else 2018
    model_name = argv[2] if len(argv) > 2 else "MLP"
    rng_seed = int(argv[3]) if len(argv) > 3 else 42
    # If seed is given, plots are not shown
    show_plots = False if len(argv) > 3 else True
    remove_outliers = True if len(argv) > 4 and argv[4].lower()=='true' else False # Remove outliers from training and test data
    check_outliers = True if len(argv) > 5 and argv[5].lower()=='true' else False # Show percentage of outliers on false negatives
    outlierType = argv[6] if len(argv) > 6 else  'intersect' # Criteria for outlier detection ('VV', 'VH', 'union')

    train_sizes = [100, 300, 500, 1000]

    grid_mean = False  # whether or not to use grid mean corrected VV and VH data
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


    # Remove outlier on entire data (train and test)
    # if remove_outliers:
    #     n_samples_SAR = X_SAR.shape[0]
    #     X_SAR, rejectedIdx = removeOutliers(X_SAR,year,outType=outlierType,returnIdx=True)
    #     X_NDVI = removeOutliers(X_NDVI,year,outType=outlierType)
    #     y_multi = removeOutliers(y_multi,year,outType=outlierType)

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
                np.random.choice(np.where(y == 1)[0], train_size//2, replace=False),
                np.random.choice(np.where(y == 0)[0], (train_size+1)//2, replace=False))
            )
            idx_test = np.delete(indices, idx_train)
            X_SAR_train, X_SAR_test = X_SAR[idx_train], X_SAR[idx_test]
            X_NDVI_train, X_NDVI_test = X_NDVI[idx_train], X_NDVI[idx_test]
            y_train, y_test = y[idx_train], y[idx_test]
        else:
            X_SAR_train, X_SAR_test, X_NDVI_train, X_NDVI_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X_SAR, X_NDVI, y, indices, train_size=train_size, random_state=rng_seed) #stratify=y)
            print(f'{(y_train==1).sum()} Colza samples on training data')

        if remove_outliers:
            n_samples_SAR = X_SAR.shape[0]
            rejectedIdx = np.full(X_SAR.shape[0], False)
            totaloutliers = countOutliers(X_SAR_train,year,idx_train,outlierType)
            while countOutliers(X_SAR_train,year,idx_train,outlierType,verbose=False) > 0:
                print('Removing outliers from training data')
                # Remove outliers from training and add to test
                X_SAR_train, rejectedIdx = removeOutliers(X_SAR_train,year,idx_train,outlierType,returnIdx=True)
                y_train = removeOutliers(y_train,year,idx_train,outlierType)
                idx_train = np.setdiff1d(idx_train, np.where(rejectedIdx))
                X_SAR_test = np.concatenate((X_SAR_test,X_SAR[rejectedIdx]),axis=0)
                y_test = np.concatenate((y_test,y[rejectedIdx]),axis=0)
                idx_test = np.concatenate((idx_test,np.where(rejectedIdx)[0]),axis=0)

                # Replace removed training data
                new_train_idx =np.random.choice(np.where(y == 1)[0], rejectedIdx.sum(), replace=False)
                idx_train = np.concatenate((idx_train,new_train_idx),axis=0)
                X_SAR_train = np.concatenate((X_SAR_train,X_SAR[new_train_idx]),axis=0)
                y_train = np.concatenate((y_train,y[new_train_idx]),axis=0)


        if check_outliers:
            countOutliers(X_SAR_train,year,outType=outlierType)


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
            # Pre_process data: rescaling
            x_test = X_SAR_test/np.percentile(X_SAR_test, 99)
            x_test[x_test > 1] = 1

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
        if remove_outliers and (totaloutliers>0): # If there were no outliers, just load standard model
            filename = filename + f'_noOutliers_{outlierType}'
        print(filename)
        y_pred = trainTestModel(model_name, path+filename, x_train, x_test,
                                y_train, y_test, dates, dates, n_epochs, batch_size)

        # Metrics
        if show_plots:
            path = "results/3_few_samples/"
            plotMetrics(y_test,  y_multi[idx_test], y_pred, path, filename)

        # Check outliers on false negatives
        if check_outliers:
            # if remove_outliers: # when removing outliers on entire data
            #     isTest = np.full(X_SAR.shape[0],False)
            #     isTest[idx_test] = True
            #     idx_test = np.full(n_samples_SAR,False)
            #     idx_test[~rejectedIdx] = isTest

            checkOutliers(y_pred,y_test,idx_test,year,outlierType)

    # print( model.parameters() )


if __name__ == "__main__":
    main(sys.argv)
