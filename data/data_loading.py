from nilearn.datasets import (
    fetch_abide_pcp,
    fetch_coords_power_2011,
)
import argparse
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiSpheresMasker
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for training the Inception_v3 model"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\Code\Data",
        help="Path where data should be downloaded or be loaded",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\Code\Out",
        help="Path to the output files",
        required=True,
    )
    parser.add_argument(
        "--fc_matrix_kind",
        type=str,
        default="tangent",
        help="different kinds of functional connectivity matrices : covariance, correlation, partial correlation, tangent, precision",
        required=False,
    )
    parser.add_argument(
        "--site_id",
        type=str,
        default="PITT",
        help="different kinds of functional connectivity matrices : covariance, correlation, partial correlation, tangent, precision",
        required=False,
    )
    args = parser.parse_args()
    return args


def load_data(
    data_dir, output_dir, fc_matrix_kind, site_id, pipeline="cpac", quality_checked=True
):

    try:  # check if feature file already exists
        # load features
        feat_file = os.path.join(
            output_dir, f"feat_matrix_{site_id}_{fc_matrix_kind}.npz"
        )
        ts_file = os.path.join(output_dir, "ABIDE_time_series.npy")
        label_file = os.path.join(output_dir, "Y_target.npz")

        correlation_matrices = np.load(feat_file)["a"]
        time_series_ls = np.load(ts_file, allow_pickle=True)
        y_target = np.load(label_file)["a"]
        print("Feature file found.")

        return correlation_matrices, time_series_ls, y_target

    except:  # if not, extract features
        print("No feature file found. Extracting features...")

        # get dataset
        print("Loading dataset...")
        abide = fetch_abide_pcp(
            data_dir=data_dir,
            pipeline=pipeline,
            quality_checked=quality_checked,
            SITE_ID=site_id,
        )

        # make list of filenames
        fmri_filenames = abide.func_preproc

        ### Previous Atlas

        # # load atlas
        # multiscale = fetch_atlas_basc_multiscale_2015()
        # # print(multiscale)
        # atlas_filename = multiscale.scale064
        # # print(f"atalas file names are: {atlas_filename}")

        # initialize masker object
        # masker = NiftiLabelsMasker(
        #     labels_img=atlas_filename, standardize=True, memory="nilearn_cache", verbose=0
        # )

        # load power atlas
        power = fetch_coords_power_2011()
        coords = np.vstack((power.rois["x"], power.rois["y"], power.rois["z"])).T

        # initialize masker object
        # NiftiSpheresMasker is useful when data from given seeds should be extracted.
        masker = NiftiSpheresMasker(
            seeds=coords,
            radius=5,  # Indicates, in millimeters, the radius for the sphere around the seed
            standardize=True,  # the signal is z-scored. Timeseries are shifted to zero mean and scaled to unit variance
            memory="nilearn_cache",
            verbose=0,
        )

        # initialize correlation measure
        correlation_measure = ConnectivityMeasure(
            kind=fc_matrix_kind, vectorize=False, discard_diagonal=True
        )

        if fc_matrix_kind == "tangent":
            time_series_ls = []
            loop = tqdm(enumerate(fmri_filenames), total=len(fmri_filenames))
            for i, sub in loop:

                # extract the timeseries from the ROIs in the atlas
                time_series = masker.fit_transform(sub)
                time_series_ls.append(time_series)
                loop.set_description(
                    f"Extracting Time series for object {i + 1} of {len(fmri_filenames)} "
                )
            correlation_matrices = correlation_measure.fit_transform(time_series_ls)

        else:
            correlation_matrices = []
            time_series_ls = []
            loop = tqdm(enumerate(fmri_filenames), total=len(fmri_filenames))
            for i, sub in loop:
                # extract the timeseries from the ROIs in the atlas
                time_series = masker.fit_transform(sub)
                # create a region x region correlation matrix
                correlation_matrix = correlation_measure.fit_transform([time_series])[0]

                time_series_ls.append(time_series)
                correlation_matrices.append(correlation_matrix)
                loop.set_description(
                    f"Extracting Time series for object {i + 1} of {len(fmri_filenames)} "
                )

        np.savez_compressed(
            os.path.join(output_dir, f"feat_matrix_{site_id}_{fc_matrix_kind}"),
            a=correlation_matrices,
            dtype=object,
        )

        np.save(
            file=os.path.join(output_dir, f"time_series_{site_id}.npy"),
            arr=time_series_ls,
            allow_pickle=True,
        )
        print(f"time series size: {len(time_series_ls)}, {time_series_ls[0].shape}")
        correlation_matrices = np.array(correlation_matrices)
        print(f"correlation matrix size: {correlation_matrices.shape}")

        # Get the target vector
        abide_pheno = pd.DataFrame(abide.phenotypic)
        y_target = abide_pheno["DX_GROUP"]
        y_target = y_target.apply(lambda x: x - 1)
        np.savez_compressed(os.path.join(output_dir, f"Y_target_{site_id}"), a=y_target)

        return correlation_matrices, time_series_ls, y_target


def run():
    args = parse_arguments()
    correlation_matrices, time_series_ls, y_target = load_data(
        args.input_path, args.output_path, args.fc_matrix_kind, site_id=args.site_id
    )
    print(f"correlation_matrices shape: {correlation_matrices.shape}")
    print(f"time_series_ls len: {len(time_series_ls)}")
    print(f"y_target shape: {y_target.shape}")

    df = pd.DataFrame(
        columns=["mean", "std", "min", "0.25", "0.5", "0.75", "0.9", "max"]
    )
    for arr in correlation_matrices:
        df2 = pd.DataFrame.from_dict(
            [
                {
                    "mean": np.mean(arr),
                    "std": np.std(arr),
                    "min": np.min(arr),
                    "0.25": np.percentile(arr, 25),
                    "0.5": np.percentile(arr, 50),
                    "0.75": np.percentile(arr, 75),
                    "0.9": np.percentile(arr, 90),
                    "max": np.max(arr),
                }
            ]
        )

        df = pd.concat([df, df2], ignore_index=True, axis=0)

    print(df)
    print(df.describe())


if __name__ == "__main__":
    run()
