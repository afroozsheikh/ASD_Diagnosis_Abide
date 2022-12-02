import argparse
import os
from tqdm import tqdm
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pickle
import torchinfo

from models.GATv2 import GATv2
from data.data_preparation import data_preparation
from utils.utils import get_metrics, count_parameters, plot_loss, plot_confusion_matrix


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for training the Inception_v3 model"
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\out\train_adjacency_tangent.npz",
        help="Path to the features file",
        required=True,
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        help="Path to the model weights file",
        required=False,
    )
    parser.add_argument(
        "--results",
        type=str,
        help="Path to the folder you want to save the model results",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Size of batch",
        required=False,
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=1,
        help="Number of heads used in Attention Mechanism",
        required=False,
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.0,
        help="dropout rate used before linear classifier",
        required=False,
    )
    parser.add_argument(
        "--last_activation",
        type=str,
        default="sigmoid",
        choices=("sigmoid", "softmax"),
        help="activation function used in the last layer. options: ['sigmoid', 'softmax']",
        required=False,
    )
    args = parser.parse_args()
    return args


def eval(model, device, dataloader, loss_fn):

    model.eval()
    y_true = []
    alllogits = []

    for batch in dataloader:

        batch = batch.to(device)
        with torch.no_grad():
            logits = model(batch)
            y_true.append(batch.y.detach().cpu())

            if model.last_activation == "sigmoid":
                logits = logits.squeeze().detach().cpu()
            else:
                logits = logits.detach().cpu()
            alllogits.append(logits)

    alllogits = torch.cat(alllogits, dim=0)
    if model.last_activation == "sigmoid":
        y_true = torch.cat(y_true, dim=0).float()
    else:
        y_true = torch.cat(y_true, dim=0).long()

    val_loss = loss_fn(alllogits, y_true)
    y_true = y_true.int()

    return val_loss.item(), alllogits, y_true


def main(args):

    tag = "GATv2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Used Device is : {}".format(device))

    with open(args.features_path, "rb") as fp:
        data_list = pickle.load(fp)

    train_data, val_data = train_test_split(
        data_list, test_size=0.2, shuffle=True, random_state=42
    )

    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    model = GATv2(
        input_feat_dim=next(iter(val_loader)).x.shape[1],
        conv_shapes=[(5, 8), (8, 16), (16, 16)],
        cls_shapes=[8],
        heads=args.heads,
        dropout_rate=args.dropout_rate,
        last_activation=args.last_activation,
    ).to(device)

    if args.weights_path is not None:
        model.load_weights(args.weights_path)
        print("Model weights loaded from the given path")

    count_parameters(model)

    if args.last_activation == "sigmoid":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    val_loss, val_logits, val_y_true = eval(model, device, val_loader, loss_fn)
    val_metrics = get_metrics(val_logits, val_y_true, args.last_activation)

    # Evaluating the best model
    print("\n\n\nModel results on validation set:")

    print(f"Validation Loss was : {val_loss:.4f}")
    print(f"Validation accuracy was : {100 * val_metrics['acc']:.2f}")
    print(f"Validation F1-score was : {100 * val_metrics['f1']:.2f}")
    print(f"Validation Precision was : {100 * val_metrics['precision']:.2f}")
    print(f"Validation Recall was : {100 * val_metrics['recall']:.2f}")

    label_names = ["0", "1"]
    plot_confusion_matrix(
        cm=val_metrics["cm"],
        classes=label_names,
        save_path=args.results,
        name="eval_cm.png",
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
