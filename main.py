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


def trainModel(model, train, n_epochs, loss_fn, optimizer, device, dates=None):
    model.train()

    for e in range(n_epochs):
        train_loss, correct = 0, 0
        labels, pred_all = [], []  # for f1-score
        for x_batch, y_batch in train:
            model.zero_grad()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Compute prediction and loss
            if model.__class__.__name__ == "LTAE_clf":
                pred = model(x_batch, dates)
            else:
                pred = model(x_batch)
            
            loss = loss_fn(pred.squeeze(), y_batch)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            labels.append(y_batch.cpu().detach().numpy())
            if pred.squeeze().ndimension() == 1:  # Binary
                correct += ((pred > 0.5).squeeze() == y_batch).sum().item()
                pred_all.append((pred > 0.5).cpu().detach().numpy())
            else:  # One-hot
                correct += (pred.argmax(1) ==
                            y_batch).type(torch.float).sum().item()
                pred_all.append((pred.argmax(1)).cpu().detach().numpy())

        labels = np.concatenate(labels, axis=0)
        pred_all = np.concatenate(pred_all, axis=0)
        f1 = f1_score(pred_all, labels, average=None)

        print(f"Epoch {e}:",
              f"Avg loss={train_loss/len(train):.4f},",
              f"Accuracy={(100*correct/len(train.dataset)):.1f}%,",
              f"F1={f1.mean():.3f} (per class {f1[0]:.2f}, {f1[1]:.2f})")

        # f1_valid = prediction(model, valid, device)
        # if f1_valid > best_validation:
        #     best_validation = f1_valid
        #     torch.save(model.state_dict(), path_file)
        #     print("\t\t BEST VALID %f" % f1_valid)

        sys.stdout.flush()


def testModel(model, test, loss_fn, device, dates=None):

    test_loss, correct = 0, 0
    labels, pred_all = [], []  # for f1-score
    with torch.no_grad():
        for x_batch, y_batch in test:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if model.__class__.__name__ == "LTAE_clf":
                pred = model(x_batch, dates)
            else:
                pred = model(x_batch)

            # Metrics
            test_loss += loss_fn(pred.squeeze(), y_batch).item()
            labels.append(y_batch.cpu().detach().numpy())
            if pred.squeeze().ndimension() == 1:  # Binary
                correct += ((pred > 0.5).squeeze() == y_batch).sum().item()
                pred_all.append((pred > 0.5).cpu().detach().numpy())
            else:  # One-hot
                correct += (pred.argmax(1) == y_batch).sum().item()
                pred_all.append((pred.argmax(1)).cpu().detach().numpy())

    labels = np.concatenate(labels, axis=0)
    pred_all = np.concatenate(pred_all, axis=0)
    f1 = f1_score(pred_all, labels, average=None)

    print(f"\nTest Error:",
          f"Avg loss={test_loss/len(test):.4f},",
          f"Accuracy={(100*correct/len(test.dataset)):.1f}%,",
          f"F1={f1.mean():.3f} (per class {f1[0]:.3f}, {f1[1]:.3f})")

    return pred_all

def trainTestModel(model_name,file_path, x_train,x_test,y_train,y_test,dates_train, dates_test=None, n_epochs=50, batch_size=16):
    n_classes = 2

    if dates_test is None:
        dates_test = dates_train

    train_dataset = TensorDataset(x_train, y_train)  # create your datset
    test_dataset = TensorDataset(x_test, y_test)  # create your datset

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2048)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Select model
    if model_name == 'MLP':
        model = MLP(x_train.shape, n_classes)
    elif model_name == 'TempCNN':
        model = TempCNN(n_classes)
    elif model_name == 'Inception':
        model = Inception(n_classes)
    elif model_name == 'LSTM-FCN':
        model = LSTMFCN(n_classes, x_train.shape[-1])
    elif model_name == 'LTAE':
        model = LTAE_clf(x_train.shape, n_classes, dates=dates_train)
    elif model_name == 'RF':
        model = RandomForestClassifier() # n_estimators=1000

    # loss = nn.BCELoss().to(device)  # requires y_train as Float not Long
    loss = nn.CrossEntropyLoss().to(device)

    # Train model (if not previously done)
    if not os.path.exists(file_path):

        if model_name == 'RF':
            model.fit(x_train.reshape(x_train.shape[0],-1), y_train)
            pickle.dump(model, open(file_path, "wb"))
        else:
            model.to(device)            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
            trainModel(model, train_dataloader, n_epochs, loss, optimizer, device, dates_train)
            torch.save(model.state_dict(), file_path)
    else:
        print(f'\n>> Loading previously-learned {model_name} model weights.\n')
        if model_name == 'RF': 
            model = pickle.load(open(file_path, "rb"))
        else:
            model.load_state_dict(torch.load(file_path))
            model.to(device)

    # Test model
    if model_name == 'RF':
        y_pred = model.predict(x_test.reshape(x_test.shape[0],-1))
        f1 = f1_score(y_pred, y_test, average=None)
        accuracy = accuracy_score(y_pred, y_test)

        print(f"\nAccuracy={(100*accuracy):.1f}%,",
                f"F1={f1.mean():.3f} (per class {f1[0]:.2f}, {f1[1]:.2f})")         
    else:    
        y_pred = testModel(model, test_dataloader, loss, device, dates_test)

    return y_pred

def main(argv):
    year = int(argv[1]) if len(argv) > 1 else 2018
    model_name = argv[2] if len(argv) > 2 else "MLP"

    n_epochs = 50

    torch.manual_seed(0)
    # np.random.seed(0)

    dataset = np.load(f'Colza_DB/Colza_data_{year}.npz', allow_pickle=True)
    X_SAR, X_NDVI, y_multi, id_parcel = dataset["X_SAR"], dataset["X_NDVI"], dataset["y"], dataset["id_parcels"]

    dates = dataset["dates_SAR"]

    # Pre-process labels: binarize with "CZH" as positive class
    y = np.zeros(y_multi.shape)
    y[y_multi == "CZH"] = 1

    # Train-test split
    indices = np.arange(y.shape[0])
    batch_size = 16
    train_size = int(batch_size * round(0.1 * X_SAR.shape[0] / batch_size))
    X_SAR_train, X_SAR_test, X_NDVI_train, X_NDVI_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_SAR, X_NDVI, y, indices, train_size=train_size, random_state=42)

    x_train = torch.Tensor(X_SAR_train)
    y_train = torch.Tensor(y_train).long()
    # y_train = F.one_hot(y_train, num_classes=n_classes).float()

    x_test = torch.Tensor(X_SAR_test)  # transform to torch tensor
    y_test = torch.FloatTensor(y_test).long()
    # y_test = F.one_hot(y_test, num_classes=n_classes).float()

    # Permute channel and time dimensions
    x_train = x_train.permute((0,2,1))
    x_test = x_test.permute((0,2,1))

    file_path = "model_weights/" + model_name + f'_SAR_{year}part_{n_epochs}ep_{x_train.shape[-2]}ch'
    y_pred = trainTestModel(model_name,file_path,x_train,x_test,y_train,y_test,dates,dates,n_epochs,batch_size)

    # Metrics
    cm = metrics.confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized, display_labels=[False, True]).plot()
    plt.show()
    # False positive breakdown
    y_test_multi = y_multi[idx_test]
    false_pos = y_test_multi[(y_pred == True) & (y_test_multi != 'CZH')]
    pd.Series(false_pos).value_counts(sort=True).plot(kind='bar')
    plt.title(f'Distribution of false positives (total of {len(false_pos)})')

    # print( model.parameters() )


if __name__ == "__main__":
    main(sys.argv)
