import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from model import MLP, TempCNN, Inception, LSTMFCN, LTAE_clf

from sklearn import metrics

from main import trainTestModel, trainModel, testModel


def main(argv):
    if len(argv) > 2:
        year_train, year_test = int(argv[1]), int(argv[2])
    else:
        year_train, year_test = 2018, 2020

    model_name = argv[3] if len(argv) > 3 else "MLP"

    grid_mean = True # whether or not to use grid mean corrected VV and VH data
    n_epochs = 50

    torch.manual_seed(0)
    # np.random.seed(0)

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


    # Pre_process data: resize
    # 2018: SAR (80288, 152, 4), NDVI (80288, 78)
    # 2019: SAR (72506, 135, 4), NDVI (72506, 80)
    # 2020: SAR (97971, 150, 4), NDVI (97971, 71)
    if model_name != 'LTAE':
        diff = X_SAR_train.shape[1] - X_SAR_test.shape[1]
        if diff > 0:
            X_SAR_train = X_SAR_train[:, (diff//2):-(diff-diff//2), :]
        elif diff < 0:
            X_SAR_test = X_SAR_test[:, -(diff//2):(diff-diff//2), :]
    # if model_name != 'LTAE':
    #     X_SAR_train = X_SAR_train[:, 1:-1, :]  # to match test data size
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
    x_test = X_SAR_test/np.percentile(X_SAR_test, 98)
    x_test[x_test > 1] = 1
    x_test = torch.Tensor(x_test)  # transform to torch tensor
    # Pre-process labels: binarize with "CZH" as positive class
    y_test = np.zeros(y_multi_test.shape)
    y_test[y_multi_test == "CZH"] = 1
    y_test = torch.Tensor(y_test).long()
    y_test = torch.Tensor(y_test).long()

    # Permute channel and time dimensions
    x_test = x_test.permute((0,2,1))

    file_path = "model_weights/" + model_name + f'_SAR_{year_train}_{n_epochs}ep_{x_train.shape[-2]}ch'
    y_pred = trainTestModel(model_name,file_path,x_train,x_test,y_train,y_test,dates_SAR_train,dates_SAR_test,n_epochs)

    # Metrics
    cm = metrics.confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized, display_labels=[False, True]).plot()
    plt.show()
    # False positive breakdown
    false_pos = y_multi_test[(y_pred == True) & (y_multi_test != 'CZH')]
    pd.Series(false_pos).value_counts(sort=True).plot(kind='bar')
    plt.title(f'Distribution of false positives (total of {len(false_pos)})')

    # print( model.parameters() )
    # exit()


if __name__ == "__main__":
    main(sys.argv)
