import argparse
import pickle
from time import time
import logging
import os
import os.path as osp
import numpy as np
import time
import torch
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torch.autograd import Variable

import random
from torch.optim.lr_scheduler import StepLR

from utils.utils_aug import stat_graph, split_class_graphs, align_graphs
from utils.utils_aug import two_graphons_mixup, universal_svd
from graphon_estimator import universal_svd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s: - %(message)s", datefmt="%Y-%m-%d"
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument(
        "--features_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\out\train_adjacency_tangent.npz",
        help="Path to the features file",
        required=True,
    )
    # parser.add_argument('--epoch', type=int, default=800)
    # parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument("--gmixup", type=str, default="False")
    parser.add_argument("--lam_range", type=str, default="[0.005, 0.01]")
    parser.add_argument("--aug_ratio", type=float, default=0.15)
    parser.add_argument("--aug_num", type=int, default=10)
    # parser.add_argument('--gnn', type=str, default="gin")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--log_screen", type=str, default="False")
    parser.add_argument("--ge", type=str, default="MC")
    args = parser.parse_args()
    return args


def prepare_dataset_x(dataset):
    if dataset[0].x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
            data.num_nodes = int(torch.max(data.edge_index)) + 1

        if max_degree < 2000:
            # dataset.transform = T.OneHotDegree(max_degree)

            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = F.one_hot(degs, num_classes=max_degree + 1).to(torch.float)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = ((degs - mean) / std).view(-1, 1)
    return dataset


def prepare_dataset_onehot_y(dataset):

    y_set = set()
    for data in dataset:
        y_set.add(int(data.y))
    num_classes = len(y_set)

    for data in dataset:
        data.y = F.one_hot(data.y, num_classes=num_classes).to(torch.float)[0]
    return dataset


# def mixup_cross_entropy_loss(input, target, size_average=True):
#     """Origin: https://github.com/moskomule/mixup.pytorch
#     in PyTorch's cross entropy, targets are expected to be labels
#     so to predict probabilities this loss is needed
#     suppose q is the target and p is the input
#     loss(p, q) = -\sum_i q_i \log p_i
#     """
#     assert input.size() == target.size()
#     assert isinstance(input, Variable) and isinstance(target, Variable)
#     loss = -torch.sum(input * target)
#     return loss / input.size()[0] if size_average else loss


# def train(model, train_loader):
#     model.train()
#     loss_all = 0
#     graph_all = 0
#     for data in train_loader:
#         # print( "data.y", data.y )
#         data = data.to(device)
#         optimizer.zero_grad()
#         output = model(data.x, data.edge_index, data.batch)
#         y = data.y.view(-1, num_classes)
#         loss = mixup_cross_entropy_loss(output, y)
#         loss.backward()
#         loss_all += loss.item() * data.num_graphs
#         graph_all += data.num_graphs
#         optimizer.step()
#     loss = loss_all / graph_all
#     return model, loss


# def test(model, loader):
#     model.eval()
#     correct = 0
#     total = 0
#     loss = 0
#     for data in loader:
#         data = data.to(device)
#         output = model(data.x, data.edge_index, data.batch)
#         pred = output.max(dim=1)[1]
#         y = data.y.view(-1, num_classes)
#         loss += mixup_cross_entropy_loss(output, y).item() * data.num_graphs
#         y = y.max(dim=1)[1]
#         correct += pred.eq(y).sum().item()
#         total += data.num_graphs
#     acc = correct / total
#     loss = loss / total
#     return (acc,)


def main(args):

    seed = args.seed
    lam_range = eval(args.lam_range)
    log_screen = eval(args.log_screen)
    gmixup = eval(args.gmixup)
    # num_epochs = args.epoch

    # num_hidden = args.num_hidden
    # batch_size = args.batch_size
    # learning_rate = args.lr
    ge = args.ge
    aug_ratio = args.aug_ratio
    aug_num = args.aug_num

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Used Device is : {}".format(device))

    with open(args.features_path, "rb") as fp:
        data_list = pickle.load(fp)

    #### Convert int to tensor in y
    for graph in data_list:
        graph.y = torch.tensor(graph.y)
        graph.y = graph.y.view(-1)

    # dataset = TUDataset(
    #     "C:\\Users\\Afrooz Sheikholeslam\\Education\\9th semester\\Project 2\\Output\\tu",
    #     name="REDDIT-BINARY",
    # )
    # dataset = list(dataset)

    # for graph in dataset:
    #     print(graph.y.view(-1), graph.y)
    #     graph.y = graph.y.view(-1)

    dataset = prepare_dataset_onehot_y(data_list)

    random.seed(seed)
    random.shuffle(dataset)

    train_nums = int(len(dataset) * 0.7)
    train_val_nums = int(len(dataset) * 0.8)

    (
        avg_num_nodes,
        avg_num_edges,
        avg_density,
        median_num_nodes,
        median_num_edges,
        median_density,
    ) = stat_graph(dataset[:train_nums])
    logger.info(f"avg num nodes of training graphs: { avg_num_nodes }")
    logger.info(f"avg num edges of training graphs: { avg_num_edges }")
    logger.info(f"avg density of training graphs: { avg_density }")
    logger.info(f"median num nodes of training graphs: { median_num_nodes }")
    logger.info(f"median num edges of training graphs: { median_num_edges }")
    logger.info(f"median density of training graphs: { median_density }")

    print(f"avg num nodes of training graphs: { avg_num_nodes }")
    print(f"avg num edges of training graphs: { avg_num_edges }")
    print(f"avg density of training graphs: { avg_density }")
    print(f"median num nodes of training graphs: { median_num_nodes }")
    print(f"median num edges of training graphs: { median_num_edges }")
    print(f"median density of training graphs: { median_density }")

    resolution = int(median_num_nodes)

    if gmixup == True:
        class_graphs = split_class_graphs(dataset[:train_nums])
        graphons = []
        for label, graphs in class_graphs:

            logger.info(f"label: {label}, num_graphs:{len(graphs)}")
            print(f"label: {label}, num_graphs:{len(graphs)}")
            align_graphs_list, normalized_node_degrees, max_num, min_num = align_graphs(
                graphs, padding=True, N=resolution
            )
            logger.info(f"aligned graph {align_graphs_list[0].shape}")
            print(f"aligned graph {align_graphs_list[0].shape}")

            logger.info(f"ge: {ge}")
            graphon = universal_svd(align_graphs_list, threshold=0.08)
            graphons.append((label, graphon))

        for label, graphon in graphons:
            logger.info(
                f"graphon info: label:{label}; mean: {graphon.mean()}, shape, {graphon.shape}"
            )
            print(
                f"graphon info: label:{label}; mean: {graphon.mean()}, shape, {graphon.shape}"
            )

        num_sample = int(train_nums * aug_ratio / aug_num)
        lam_list = np.random.uniform(
            low=lam_range[0], high=lam_range[1], size=(aug_num,)
        )

        random.seed(seed)
        new_graph = []
        for lam in lam_list:
            logger.info(f"lam: {lam}")
            logger.info(f"num_sample: {num_sample}")
            print(f"lam: {lam}")
            print(f"num_sample: {num_sample}")
            two_graphons = random.sample(graphons, 2)
            new_graph += two_graphons_mixup(two_graphons, la=lam, num_sample=num_sample)
            logger.info(f"label: {new_graph[-1].y}")
            print(f"label: {new_graph[-1].y}")

        (
            avg_num_nodes,
            avg_num_edges,
            avg_density,
            median_num_nodes,
            median_num_edges,
            median_density,
        ) = stat_graph(new_graph)
        logger.info(f"avg num nodes of new graphs: { avg_num_nodes }")
        logger.info(f"avg num edges of new graphs: { avg_num_edges }")
        logger.info(f"avg density of new graphs: { avg_density }")
        logger.info(f"median num nodes of new graphs: { median_num_nodes }")
        logger.info(f"median num edges of new graphs: { median_num_edges }")
        logger.info(f"median density of new graphs: { median_density }")

        print(f"avg num nodes of new graphs: { avg_num_nodes }")
        print(f"avg num edges of new graphs: { avg_num_edges }")
        print(f"avg density of new graphs: { avg_density }")
        print(f"median num nodes of new graphs: { median_num_nodes }")
        print(f"median num edges of new graphs: { median_num_edges }")
        print(f"median density of new graphs: { median_density }")

        dataset = new_graph + dataset
        logger.info(f"real aug ratio: {len( new_graph ) / train_nums }")
        print(f"real aug ratio: {len( new_graph ) / train_nums }")
        train_nums = train_nums + len(new_graph)
        train_val_nums = train_val_nums + len(new_graph)

    dataset = prepare_dataset_x(dataset)

    logger.info(f"num_features: {dataset[0].x.shape}")
    logger.info(f"num_classes: {dataset[0].y.shape}")
    print(f"num_features: {dataset[0].x.shape}")
    print(f"num_classes: {dataset[0].y.shape}")

    num_features = dataset[0].x.shape[1]
    num_classes = dataset[0].y.shape[0]

    train_dataset = dataset[:train_nums]
    random.shuffle(train_dataset)
    val_dataset = dataset[train_nums:train_val_nums]
    test_dataset = dataset[train_val_nums:]

    logger.info(f"train_dataset size: {len(train_dataset)}")
    logger.info(f"val_dataset size: {len(val_dataset)}")
    logger.info(f"test_dataset size: {len(test_dataset)}")

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # filename = f"augmented_{args.features_path}"
    # path = os.path.join(args.output_path, filename)
    with open(f"{args.features_path}_augmented", "wb") as fp:
        pickle.dump(dataset, fp)
    print(f"Features are successfully extracted and stored in path")
    print("******", len(dataset))

    # if model == "GIN":
    #     model = GIN(
    #         num_features=num_features, num_classes=num_classes, num_hidden=num_hidden
    #     ).to(device)
    # else:
    #     logger.info(f"No model.")

    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=learning_rate, weight_decay=5e-4
    # )
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    # for epoch in range(1, num_epochs):
    #     model, train_loss = train(model, train_loader)
    #     train_acc = 0
    #     val_acc, val_loss = test(model, val_loader)
    #     test_acc, test_loss = test(model, test_loader)
    #     scheduler.step()

    #     logger.info(
    #         "Epoch: {:03d}, Train Loss: {:.6f}, Val Loss: {:.6f}, Test Loss: {:.6f},  Val Acc: {: .6f}, Test Acc: {: .6f}".format(
    #             epoch, train_loss, val_loss, test_loss, val_acc, test_acc
    #         )
    #     )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
