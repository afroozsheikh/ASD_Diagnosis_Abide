import torch
import os
from sklearn import metrics
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import itertools
import numpy as np


def get_metrics(logits, y_true, last_activation):

    if last_activation == "sigmoid":
        y_pred = torch.sigmoid(logits)
        y_pred = (y_pred >= 0.5).int()
    else:
        y_pred = torch.softmax(logits, dim=-1)
        y_pred = torch.argmax(y_pred, dim=-1)

    metrics_dict = {}

    metrics_dict["acc"] = metrics.accuracy_score(y_pred, y_true)
    metrics_dict["f1"] = metrics.f1_score(y_pred, y_true, zero_division=0)
    metrics_dict["precision"] = metrics.precision_score(y_pred, y_true, zero_division=0)
    metrics_dict["recall"] = metrics.recall_score(y_pred, y_true, zero_division=0)
    metrics_dict["cm"] = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)

    return metrics_dict


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters shape", "Total Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_shape = parameter.shape
            params = parameter.numel()
            table.add_row([name, param_shape, params])
            total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def plot_loss(train_losses, val_losses, save_path, name="loss.png"):

    plt.plot(train_losses, "b", label="train_loss")
    plt.plot(val_losses, "r", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title("Traing and Validation losses curve")
    plt.savefig(os.path.join(save_path, name))
    plt.show()


def plot_confusion_matrix(
    cm,
    classes,
    save_path,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    name="cm.png",
):

    plt.imshow(cm, interpolation="nearest", cmap=cmap, aspect="auto")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join(save_path, name))
    plt.show()
