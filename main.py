
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from model import MLP  # , TempCNN


def trainModel(model, train, n_epochs, loss_fn, optimizer, device):
    model.train()

    for e in range(n_epochs):
        train_loss, correct = 0, 0
        labels, pred_all = [], []  # for f1-score
        for x_batch, y_batch in train:
            model.zero_grad()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Compute prediction and loss
            pred = model(x_batch)
            loss = loss_fn(pred.squeeze(), y_batch)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            correct += (pred.argmax(1) ==
                        y_batch).type(torch.float).sum().item()
            labels.append(y_batch)  # .detach().numpy()
            pred_all.append(pred.argmax(1))

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


def testModel(model, test, loss_fn):

    test_loss, correct = 0, 0
    labels, pred_all = [], []  # for f1-score
    with torch.no_grad():
        for x_batch, y_batch in test:
            pred = model(x_batch)
            test_loss += loss_fn(pred.squeeze(), y_batch).item()
            correct += (pred.argmax(1) ==
                        y_batch).type(torch.float).sum().item()
            labels.append(y_batch)  # .detach().numpy()
            pred_all.append(pred.argmax(1))

    labels = np.concatenate(labels, axis=0)
    pred_all = np.concatenate(pred_all, axis=0)
    f1 = f1_score(pred_all, labels, average=None)

    print(f"\nTest Error:",
          f"Avg loss={test_loss/len(test):.4f},",
          f"Accuracy={(100*correct/len(test.dataset)):.1f}%,",
          f"F1={f1.mean():.3f} (per class {f1[0]:.2f}, {f1[1]:.2f})")


def main(argv):
    year = int(argv[1])
    n_classes = 2
    n_epochs = 100

    torch.manual_seed(0)
    # np.random.seed(0)

    dataset = np.load(f'Colza_DB/Colza_data_{year}.npz', allow_pickle=True)
    X_SAR, X_NDVI, y_multi, id_parcel = dataset["X_SAR"], dataset["X_NDVI"], dataset["y"], dataset["id_parcels"]

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

    train_dataset = TensorDataset(x_train, y_train)  # create your datset
    test_dataset = TensorDataset(x_test, y_test)  # create your datset

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2048)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MLP(np.prod(x_train.shape[1:]), n_classes, dropout_rate=.5)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-5, weight_decay=1e-6)

    # loss = nn.BCELoss().to(device) # requires y_train as Float not Long
    loss = nn.CrossEntropyLoss().to(device)

    trainModel(model, train_dataloader, n_epochs, loss, optimizer, device)

    testModel(model, test_dataloader, loss)

    torch.save(model.state_dict(), 'MLP_weights')

    # print( model.parameters() )
    # exit()


if __name__ == "__main__":
    main(sys.argv)
