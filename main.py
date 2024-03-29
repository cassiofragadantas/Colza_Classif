import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib.ticker import PercentFormatter
import seaborn as sns


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

            loss = loss_fn(pred, y_batch)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            labels.append(y_batch.cpu().detach().numpy())
            if pred.shape[1] == 1:  # Binary
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
              f"Accuracy={(100*correct/len(train.dataset)):.3f}%,",
              f"F1={f1.mean()*100:.3f} (per class {f1[0]*100:.3f}, {f1[1]*100:.3f})")

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
          f"Avg loss={test_loss/len(test):.4f},")

    # printMeasures(pred_all,labels)

    return pred_all


def trainTestModel(model_name, file_path, x_train, x_test, y_train, y_test, dates_train, dates_test=None, n_epochs=50, batch_size=16):
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
        model = RandomForestClassifier()  # n_estimators=1000

    # loss = nn.BCELoss().to(device)  # requires y_train as Float not Long
    loss = nn.CrossEntropyLoss().to(device)

    # Train model (if not previously done)
    if not os.path.exists(file_path):
        print(f'\n>> Training {model_name} model on {device}.\n')
        start_time = time.time()
        if model_name == 'RF':
            model.fit(x_train.reshape(x_train.shape[0], -1), y_train)
            pickle.dump(model, open(file_path, "wb"))
        else:
            model.to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=1e-5, weight_decay=1e-6)
            trainModel(model, train_dataloader, n_epochs,
                       loss, optimizer, device, dates_train)
            torch.save(model.state_dict(), file_path)
        print(f'\nTraining time = {time.time()-start_time:.2f} seconds')
    else:
        print(f'\n>> Loading previously-learned {model_name} model weights.\n')
        if model_name == 'RF':
            model = pickle.load(open(file_path, "rb"))
        else:
            model.load_state_dict(torch.load(file_path, map_location=device))
            model.to(device)

    # Test model
    start_time = time.time()
    if model_name == 'RF':
        y_pred = model.predict(x_test.reshape(x_test.shape[0], -1))
    else:
        y_pred = testModel(model, test_dataloader, loss, device, dates_test)
    print(f'\nTesting time = {time.time()-start_time:.2f} seconds\n')

    # Performance measures
    printMeasures(y_pred,y_test)

    return y_pred

def printMeasures(y_pred,y_real,verbose=True):
    # Accuracy - F1
    accuracy = accuracy_score(y_pred, y_real)
    f1 = f1_score(y_pred, y_real, average=None)
    print(f"Accuracy={(100*accuracy):.3f}%,",
          f"F1={f1.mean()*100:.3f} (per class {f1[0]*100:.3f}, {f1[1]*100:.3f})")
    # Precision - recall -  kappa - confusion matrix
    if verbose:
        precision, recall, f1, _ = precision_recall_fscore_support(y_real,y_pred)
        kappa = cohen_kappa_score(y_real, y_pred)
        print(f"Precision={100*precision[1]:.3f}%,",
            f"Recall={100*recall[1]:.3f}%, Kappa={kappa:.4f}") # Positive class only - standard
        print(f"\nOther (non-standard for binary classif):")        
        print(f"Non-colza: Precision={100*precision[0]:.3f}%,",
            f"Recall={100*recall[0]:.3f}%")
        print(f"Overall (average): Precision={100*precision.mean():.3f}%,",
            f"Recall={100*recall.mean():.3f}%")
        cm = metrics.confusion_matrix(y_real, y_pred)
        print("\nConfusion matrix:")
        print(f"[TN, FP] = [{cm[0,0]:5d}, {cm[0,1]:5d}]\n[FN, TP]   [{cm[1,0]:5d}, {cm[1,1]:5d}]") #{' ':.5s}  
    

def checkOutliers(y_pred,y_test,idx_test,year,outType='intersect'):
    if not os.path.exists(f'Colza_DB/Outliers_{year}.npz'):
        raise ValueError('Outliers file is missing. Please run outlier_detection.py first')
    
    outliers = np.load(f'Colza_DB/Outliers_{year}.npz')

    FN = (y_pred==False) & (y_test.cpu().detach().numpy()==True) # False negatives

    # Check intersection between false negatives and outliers
    print('\nOUTLIERS ANALYSIS (positive class):')
    for outType_k in outliers.files:
        out = outliers[outType_k][idx_test]
        corr =  FN & out
        print(f'{100*corr.sum()/out.sum():.1f}% ({corr.sum()}/{out.sum()}) outliers are false negatives. ' +
              f'{100*corr.sum()/FN.sum():.1f}% ({corr.sum()}/{FN.sum()}) false negatives are outliers ({outType_k})')

    print('\nPerformance excluding outliers on test:')
    out = outliers[outType][idx_test]
    printMeasures(y_pred[~out],y_test[~out])


def countOutliers(data,year,idx=None,outType='intersect',verbose=True):
    if not os.path.exists(f'Colza_DB/Outliers_{year}.npz'):
        raise ValueError('Outliers file is missing. Please run outlier_detection.py first')

    if idx is None:
        idx = range(data.shape[0])
  
    outliers = np.load(f'Colza_DB/Outliers_{year}.npz')
    outliers = outliers[outType][idx]
    if verbose:
        print(f'\nTotal of {outliers.sum()} outliers found in the training data.')
    return outliers.sum()

def removeOutliers(data,year,idx=None,outType='intersect',returnIdx=False):
    #TODO should return new idx after removals, otherwise doesn't work idx different from whole dataset
    if not os.path.exists(f'Colza_DB/Outliers_{year}.npz'):
        raise ValueError('Outliers file is missing. Please run outlier_detection.py first')

    if idx is None:
        idx = range(data.shape[0])

    outliers = np.load(f'Colza_DB/Outliers_{year}.npz')
    outliersIdx = outliers[outType][idx]

    if returnIdx:
        rejectedIdx = np.full(len(outliers[outType]), False)
        rejectedIdx[idx] = outliersIdx
        return data[~outliersIdx], rejectedIdx
    else:
        return data[~outliersIdx]

def plotMetrics(y_test, y_multi_test, y_pred,path,filename):
    # Confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    # metrics.ConfusionMatrixDisplay(
    #     confusion_matrix=cm_normalized, display_labels=[False, True]).plot()
    plotFullConfusionMatrix(cm, cm_normalized,path,filename)
    plt.show()
    # False positive breakdown
    false_pos = y_multi_test[(y_pred == True) & (y_multi_test != 'CZH')]
    pd.Series(false_pos).value_counts(sort=True).plot(kind='bar')
    plt.title(f'Distribution of false positives (total of {len(false_pos)})')
    plt.show()


def plotFullConfusionMatrix(cm, cm_norm, path, filename, figsize=(2.1, 2.1)):
    classes = ['False', 'True']
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            annot[i, j] = '%.1f%%\n(%d)' % (cm_norm[i, j]*100, cm[i, j])
    cm_norm = pd.DataFrame(cm_norm)
    cm_norm = cm_norm * 100
    cm_norm.index.name = 'True label'
    cm_norm.columns.name = 'Predicted label'
    _, ax = plt.subplots(figsize=figsize)
    plt.yticks(va='center')
    # plt.title(title)

    sns.heatmap(cm_norm, annot=annot, fmt='', ax=ax, xticklabels=classes, cbar=False,
                cbar_kws={'format': PercentFormatter()}, yticklabels=classes, cmap="Blues")

    # Save figure
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path+filename, bbox_inches="tight")


def main(argv):
    year = int(argv[1]) if len(argv) > 1 else 2018
    model_name = argv[2] if len(argv) > 2 else "MLP"
    show_plots = False if len(argv) > 3 else True

    rng_seed = 42
    grid_mean = True  # whether or not to use grid mean corrected VV and VH data
    n_epochs = 100

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
    indices = np.arange(y.shape[0])
    batch_size = 16
    ratio = 0.7  # train / (train + test)
    train_size = int(batch_size * round(ratio * X_SAR.shape[0] / batch_size))
    X_SAR_train, X_SAR_test, X_NDVI_train, X_NDVI_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_SAR, X_NDVI, y, indices, train_size=train_size, random_state=rng_seed)

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
    filename = model_name+f'_SAR_{year}part_{n_epochs}ep_{x_train.shape[-2]}ch'
    y_pred = trainTestModel(model_name, path+filename, x_train, x_test,
                            y_train, y_test, dates, dates, n_epochs, batch_size)

    # Metrics
    if show_plots:
        path = "results/1_same_year/"
        plotMetrics(y_test,  y_multi[idx_test], y_pred,path,filename)

    # print( model.parameters() )


if __name__ == "__main__":
    main(sys.argv)
