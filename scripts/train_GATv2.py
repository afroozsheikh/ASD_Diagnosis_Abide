import argparse
import pickle
import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch_geometric.loader import DataLoader
import torchinfo
from torch.autograd import Variable

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
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model",
        required=True,
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=1,
        help="Number of heads used in Attention Mechanism",
        required=False,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning Rate used for training the model",
        required=False,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Earlt Stopping patience",
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
    parser.add_argument(
        "--is_augmented",
        type=str,
        default="True",
        choices=("True", "False"),
        required=False,
    )
    args = parser.parse_args()
    return args


def mixup_cross_entropy_loss(input, target, size_average=True):
    """Origin: https://github.com/moskomule/mixup.pytorch
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)
    loss = -torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss


def train(model, device, batch, optimizer, loss_fn, args):

    model.train()
    batch = batch.to(device)
    optimizer.zero_grad()
    logits = model(batch)
    if args.is_augmented == "True":  ## softmax should be used
        y = batch.y.view(-1, model.num_class)
        loss = mixup_cross_entropy_loss(logits, y)

    else:
        if model.last_activation == "sigmoid":
            loss = loss_fn(logits.squeeze(), batch.y.float())
        else:
            loss = loss_fn(logits, batch.y.long())

    loss.backward()
    optimizer.step()

    return loss.item()


def eval_batch(model, device, batch, args):

    model.eval()
    batch = batch.to(device)

    with torch.no_grad():
        logits = model(batch)

    y_true = batch.y.detach().cpu()
    if args.is_augmented == "True":
        y_true = batch.y.view(-1, model.num_class)

    if model.last_activation == "sigmoid":
        logits = logits.squeeze().detach().cpu()
    else:
        logits = logits.detach().cpu()

    return logits, y_true


def eval(model, device, dataloader, loss_fn, args):
    alllogits = []
    y_true = []
    model.eval()
    if args.is_augmented == "True":
        val_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            logits = model(batch)
            alllogits.append(logits)
            y = batch.y.view(-1, model.num_class)
            y_true.append(y)
            val_loss += mixup_cross_entropy_loss(logits, y).item() * batch.num_graphs
        y_true = torch.cat(y_true, dim=0).long()
    else:
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
        if model.last_activation == "sigmoid":
            y_true = torch.cat(y_true, dim=0).float()
        else:
            y_true = torch.cat(y_true, dim=0).long()

        val_loss = loss_fn(alllogits, y_true).item()
        y_true = y_true.int()

    alllogits = torch.cat(alllogits, dim=0)
    return val_loss, alllogits, y_true


def main(args):

    tag = "GATv2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Used Device is : {}".format(device))

    with open(args.features_path, "rb") as fp:
        data_list = pickle.load(fp)

    train_data, val_data = train_test_split(
        data_list, test_size=0.2, shuffle=True, random_state=42
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    model = GATv2(
        input_feat_dim=next(iter(train_loader)).x.shape[1],
        conv_shapes=[(5, 8), (8, 16), (16, 16), (16, 32)],
        cls_shapes=[16, 8],
        heads=args.heads,
        dropout_rate=args.dropout_rate,
        last_activation=args.last_activation,
    ).to(device)

    if args.weights_path is not None:
        model.load_weights(args.weights_path)
        print("Model weights loaded from the given path")

    count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.last_activation == "sigmoid":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    best_val_loss = 1000
    best_metrics = {}
    trigger_times = 0

    for epoch in range(1, 1 + args.epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, batch in loop:

            loss = train(model, device, batch, optimizer, loss_fn, args)

            logits, y_true = eval_batch(model, device, batch, args)
            train_metrics_batch = get_metrics(
                logits, y_true, args.last_activation, args
            )

            loop.set_description(f"Epoch: {epoch:02d}/{args.epochs:02d}")
            loop.set_postfix_str(
                f"batch {batch_idx+1}/{len(train_loader)}, "
                f"Loss: {loss:.4f}, "
                f"Accuracy: {100 * train_metrics_batch['acc']:.2f}%"
            )

        loss, train_logits, train_y_true = eval(
            model, device, train_loader, loss_fn, args
        )
        train_metrics = get_metrics(
            train_logits, train_y_true, args.last_activation, args
        )

        val_loss, val_logits, val_y_true = eval(
            model, device, val_loader, loss_fn, args
        )
        val_metrics = get_metrics(val_logits, val_y_true, args.last_activation, args)

        train_losses.append(loss)
        val_losses.append(val_loss)

        print(
            f"Loss: {loss:.4f}, "
            f"Accuracy: {100 * train_metrics['acc']:.2f}%, "
            f"Val_Loss: {val_loss:.4f}, "
            f"Val_Accuracy: {100 * val_metrics['acc']:.2f}%"
        )

        # Early stopping
        if val_loss >= best_val_loss:
            trigger_times += 1
            print(
                f"Val_Loss didn't improve from {best_val_loss}, trigger_times is {trigger_times}"
            )

            if trigger_times >= args.patience:
                print("Early stopping reached trigger_times limit")
                break

        else:
            print(f"Val_Loss improved from {best_val_loss} to {val_loss}")
            trigger_times = 0

        # Model Checkpoint
        if val_loss < best_val_loss:
            weights_path = os.path.join(args.results, f"model_weights_{tag}.pt")
            torch.save(model.state_dict(), weights_path)
            best_val_loss = val_loss
            best_metrics = val_metrics
            print(f"Model Checkpointed: model saved in {weights_path}")

    # Evaluating the best model
    print("\n\n\nBest Model results on validation set:")

    print(f"Best Validation Loss was : {best_val_loss:.4f}")
    print(f"Best Validation accuracy was : {100 * best_metrics['acc']:.2f}")
    print(f"Best Validation F1-score was : {100 * best_metrics['f1']:.2f}")
    print(f"Best Validation Precision was : {100 * best_metrics['precision']:.2f}")
    print(f"Best Validation Recall was : {100 * best_metrics['recall']:.2f}")

    plot_loss(train_losses=train_losses, val_losses=val_losses, save_path=args.results)
    label_names = ["0", "1"]
    plot_confusion_matrix(
        cm=best_metrics["cm"], classes=label_names, save_path=args.results
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
