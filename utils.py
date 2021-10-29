import torch
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def gen_A(adj_file):
    result = pickle.load(open(adj_file, 'rb'))
    # convert to [0, 1] by taking into account neg correlation
    _adj = np.abs(result.to_numpy())
    for i in range(7):
        for j in range(7):
            if i != j:
                _adj[i, j] = 0
    # apply only down threshold
    _adj[_adj < 0.1] = 0
    _adj[_adj >= 0.1] = 1
    _adj = _adj * 0.5 / (_adj.sum(1, keepdims=True) - 1)
    for i in range(9):
        _adj[i, i] = 0.5
    return _adj


def CCC_metric(outputs, labels):
    mean_labels = torch.mean(labels, axis=0)
    mean_outputs = torch.mean(outputs, axis=0)
    var_labels = torch.var(labels, axis=0)
    var_outputs = torch.var(outputs, axis=0)
    cor = torch.mean((outputs - mean_outputs) * (labels - mean_labels), axis=0)
    r = 2*cor / (var_labels + var_outputs + (mean_labels-mean_outputs)**2)
    return r


def save_cm_plot(cm, target_names, output_name, title='Confusion matrix', cmap=None, normalize=True):
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))

    plt.savefig(output_name)



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']