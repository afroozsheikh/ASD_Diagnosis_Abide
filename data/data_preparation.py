import numpy as np
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from networkx.algorithms.centrality import betweenness_centrality
from networkx.algorithms.cluster import clustering
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import skew, kurtosis
import pickle
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Arguments for feature extraction")
    parser.add_argument(
        "--adj_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\Code\Out\ABIDE_adjacency.npz",
        help="Path to the adjacancy matrix",
        required=True,
    )
    parser.add_argument(
        "--time_series_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\out\time_series.npy",
        help="Path to the time series matrix",
        required=True,
    )
    parser.add_argument(
        "--y_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\Code\Out\Y_target.npz",
        help="Path to the y target",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\Code\Out",
        help="Path to the output folder you want to save the features in",
        required=True,
    )
    parser.add_argument(
        "--adj_mat_type",
        type=str,
        default="weighted_threshold",
        choices=("Weighted", "weighted_threshold", "binary_threshold"),
        help="Method used for making the adjacency matrix. options: ['Weighted', 'weighted_threshold', 'binary_threshold']",
        required=False,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold",
        required=False,
    )
    parser.add_argument(
        "--data_scaler_type",
        type=str,
        default=None,
        choices=("MinMax", "Standard"),
        help="Method used for scaling the data. options: ['MinMax', 'Standard']",
        required=False,
    )
    parser.add_argument(
        "--loop_removal",
        type=str,
        default="True",
        choices=("True", "False"),
        help="If true, then the loops will be removed",
        required=False,
    )
    parser.add_argument(
        "--abide_path",
        type=str,
        default="True",
        choices=("True", "False"),
        help="If true, then the loops will be removed",
        required=False,
    )

    args = parser.parse_args()
    return args


def get_groups(abide_path):
    abide = datasets.fetch_abide_pcp(
        data_dir=abide_path,
        pipeline="cpac",
        quality_checked=True,
    )

    # Transform phenotypic data into dataframe
    abide_pheno = pd.DataFrame(abide.phenotypic)

    # Extract group info
    groups = []
    for s in abide_pheno.SITE_ID:
        groups.append(s.decode())  # for some reason the site names are of type 'bytes'


def show_data_statistics(datalist):
    normal_dfs = []
    ASD_dfs = []
    for subject in data_list:
        if subject.y == 0:
            df = pd.DataFrame(
                data=subject.x.numpy(),
                columns=[
                    "degree",
                    "betweenness",
                    "eccentricity",
                    "ts_mean",
                    "ts_variance",
                    # "ts_skewness",
                    # "ts_kurtosis",
                ],
            )
            normal_dfs.append(df)
        else:
            df = pd.DataFrame(
                data=subject.x.numpy(),
                columns=[
                    "degree",
                    "betweenness",
                    "eccentricity",
                    "ts_mean",
                    "ts_variance",
                    # "ts_skewness",
                    # "ts_kurtosis",
                ],
            )
            ASD_dfs.append(df)
    return normal_dfs, ASD_dfs


def data_preparation(
    abide_path,
    adj_path,
    time_series_path,
    y_path,
    output_path,
    adj_mat_type="weighted_threshold",
    threshold=0.2,
    scaler_type=None,
    loop_removal="True",
):
    """
    Data Preparation with Leave One Group Out split
    Creates Data object of pytorch_geometric using graph features and edge list

    Args:
        adj_path (str): path to the adjacancy matrix (.npz)
        time_series_path (str): path to the time_series matrix (.npy)
        y_path (str): path to the label matrix (.npz)
        threshold (float, optional): threshold used to remove noisy connections from adj_mat. Defaults to 0.2.

    Returns:
        tuple: train, validation, test dataloaders
    """

    idx1 = adj_path.rfind("_") + 1
    idx2 = adj_path.rfind(".")
    fc_matrix_kind = adj_path[idx1:idx2]
    filename = (
        f"features_{fc_matrix_kind}_{adj_mat_type}_{str(threshold)}_{scaler_type}_3"
    )
    min_edge = 10000
    try:  # check if feature file already exists

        # load features
        feat_file = os.path.join(output_path, filename)

        with open(feat_file, "rb") as fp:
            data_list = pickle.load(fp)

        print("Feature file found.")
        normal_dfs, ASD_dfs = show_data_statistics(datalist)

    except:  # if not, extract features
        print("No feature file found. Extracting features...")
        input_p = os.path.join(output_path, filename)
        adj_mat = np.load(adj_path)["a"]
        adj_mat = np.load(adj_path)["a"]
        time_series_ls = np.load(time_series_path, allow_pickle=True)
        y_target = np.load(y_path)["a"]

        if adj_mat_type not in ["Weighted", "weighted_threshold", "binary_threshold"]:
            raise RuntimeError(
                "adj_mat_type should be one of these: ['Weighted', 'weighted_threshold', 'binary_threshold']"
            )
        else:
            print("binaryyy")
            if adj_mat_type == "weighted_threshold":
                adj_mat[adj_mat <= threshold] = 0
            elif adj_mat_type == "binary_threshold":
                adj_mat[adj_mat <= threshold] = 0
                adj_mat[adj_mat >= threshold] = 1
            else:
                pass

        data_list = []
        ## Create a graph using networkx
        loop = tqdm(range(adj_mat.shape[0]), total=adj_mat.shape[0])
        for i in loop:

            loop.set_description(
                f"Extracting features of object {i} of {adj_mat.shape[0]} "
            )

            if loop_removal == "True":
                np.fill_diagonal(adj_mat[i], 0)

            G = nx.from_numpy_matrix(adj_mat[i], create_using=nx.Graph)
            print(f"number of edges: {G.number_of_edges()}")
            if G.number_of_edges() < min_edge:
                min_edge = G.number_of_edges()

            ## Extract features

            ## dict(G.degree(weight="weight")).values()
            ## dict(betweenness_centrality(G, weight="weight")).values()
            if not nx.is_connected(G):
                eccentricity = {}
                components = sorted(nx.connected_components(G), key=len, reverse=True)
                for comp in components:
                    G_sub = G.subgraph(comp)
                    for node in comp:
                        eccentricity[node] = nx.eccentricity(G_sub, v=node)

            features = pd.DataFrame(
                {
                    "degree": dict(G.degree(weight="weight")).values(),
                    "betweenness": dict(
                        betweenness_centrality(G, weight="weight")
                    ).values(),
                    "eccentricity": dict(nx.eccentricity(G)).values()
                    if nx.is_connected(G) == True
                    else eccentricity.values(),
                    "ts_mean": time_series_ls[i].mean(axis=0),
                    "ts_variance": time_series_ls[i].var(axis=0),
                    # "ts_skewness": skew(time_series_ls[i], axis=0),
                    # "ts_kurtosis": kurtosis(time_series_ls[i], axis=0),
                }
            )
            normal_dfs, ASD_dfs = show_data_statistics(datalist)

            # scale the data (optional)
            if scaler_type in ["MinMax", "Standard"]:
                if scaler_type == "MinMax":
                    scaler = MinMaxScaler()
                    features = scaler.fit_transform(features)
                else:
                    scaler = StandardScaler()
                    features = scaler.fit_transform(features)

                X = torch.from_numpy(features)
            else:
                X = torch.tensor(features.values)

            edge_index = torch.tensor(list(G.edges()))
            data_list.append(Data(x=X, edge_index=edge_index.T, y=y_target[i].item()))

            # save features
        path = os.path.join(output_path, filename)
        with open(path, "wb") as fp:
            pickle.dump(data_list, fp)
        print(f"Features are successfully extracted and stored in: {path}")
        print(min_edge)
        return normal_dfs, ASD_dfs


def main():
    args = parse_arguments()

    normal_dfs, ASD_dfs = data_preparation(
        abide_path=args.abide_path,
        adj_path=args.adj_path,
        time_series_path=args.time_series_path,
        y_path=args.y_path,
        output_path=args.output_path,
        adj_mat_type=args.adj_mat_type,
        threshold=args.threshold,
        scaler_type=args.data_scaler_type,
        loop_removal=args.loop_removal,
    )

    print("Normal subject:")
    print(normal_dfs[0].describe())
    print()
    print("Subject with ASD:")
    print(ASD_dfs[1].describe())


if __name__ == "__main__":
    main()
