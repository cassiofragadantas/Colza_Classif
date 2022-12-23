import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from model import MLP  # , TempCNN

from sklearn import metrics

from main import trainModel, testModel


def main(argv):
    if len(argv) > 2:
        year_train, year_test = int(argv[1]), int(argv[2])
    else:
        year_train, year_test = 2018, 2020

    n_classes = 2
    n_epochs = 20

    torch.manual_seed(0)
    # np.random.seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training data
    dataset = np.load(
        f'Colza_DB/Colza_data_{year_train}.npz', allow_pickle=True)
    X_SAR, X_NDVI, y_multi_train = dataset["X_SAR"], dataset["X_NDVI"], dataset["y"]
    id_parcels_train, dates_SAR_train, dates_NDVI_train = \
        dataset["id_parcels"], dataset["dates_SAR"], dataset["dates_NDVI"]

    # Pre_process data: rescaling
    x_train = X_SAR[:, 1:-1, :]  # to match test data size
    x_train = x_train/np.percentile(x_train, 99)
    x_train[x_train > 1] = 1
    x_train = torch.Tensor(x_train)
    # Pre-process labels: binarize with "CZH" as positive class
    y_train = np.zeros(y_multi_train.shape)
    y_train[y_multi_train == "CZH"] = 1
    y_train = torch.Tensor(y_train).long()

    train_dataset = TensorDataset(x_train, y_train)  # create your datset
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)

    # Test data
    dataset = np.load(
        f'Colza_DB/Colza_data_{year_test}.npz', allow_pickle=True)
    X_SAR, X_NDVI, y_multi_test = dataset["X_SAR"], dataset["X_NDVI"], dataset["y"]
    id_parcel_test, dates_SAR_test, dates_NDVI_test = \
        dataset["id_parcels"], dataset["dates_SAR"], dataset["dates_NDVI"]

    # Pre_process data: rescaling
    x_test = X_SAR/np.percentile(X_SAR, 98)
    x_test[x_test > 1] = 1
    x_test = torch.Tensor(x_test)  # transform to torch tensor
    # Pre-process labels: binarize with "CZH" as positive class
    y_test = np.zeros(y_multi_test.shape)
    y_test[y_multi_test == "CZH"] = 1
    y_test = torch.Tensor(y_test).long()
    y_test = torch.Tensor(y_test).long()

    test_dataset = TensorDataset(x_test, y_test)  # create your datset
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2048)

    # Train and test model
    model = MLP(np.prod(x_train.shape[1:]), n_classes, dropout_rate=.5)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-5, weight_decay=1e-6)

    # loss = nn.BCELoss().to(device)  # requires y_train as Float not Long
    loss = nn.CrossEntropyLoss().to(device)

    trainModel(model, train_dataloader, n_epochs, loss, optimizer, device)

    y_pred = testModel(model, test_dataloader, loss)

    torch.save(model.state_dict(), 'temporal_transfer_MLP_weights')

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
