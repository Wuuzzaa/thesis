import warnings

import pandas as pd

from calc_scores import get_X_train_X_test_y_train_y_test
from kmeans_feature import _create_kmeans_features
from lda_feature import _create_lda_features
from pca_feature import _create_pca_features
from umap_feature import _create_umap_features
from util import print_function_header, get_sub_folders
from pathlib import Path
from tqdm import tqdm
from time import time


def create_features(
        feature_type: str,
        train_filename: str,
        test_filename: str,
        datasets_folder: Path,
        transformer_params: dict,
        prefix: str,
        random_state: int,
        X_file_name: str,
        y_file_name: str,
        path_results_file: Path,
        pca_mode: str = None,
        kmeans_n_cluster_range: range = None,
        umap_range_n_components: range = None,
):
    # set the feature_name_column accoring to the type of feature created and the kind of train data like cleaned or filtered
    feature_type_column_name = feature_type

    # we must have distinct between pca and kpca
    if feature_type == "pca":
        feature_type_column_name = pca_mode

    # now distinct between cleaned and cleaned filtered data
    if "_filtered." in X_file_name:
        feature_type_column_name += "_filtered"

    elif "_clean." in X_file_name:
        feature_type_column_name += "_clean"

    else:
        raise NotImplemented(f"Not clean not clean_filtered data X_file")

    # load results dataframe
    results_df = pd.read_feather(path_results_file)

    # skip if done
    if f"{feature_type_column_name}_n_features_created" in results_df.columns and \
            f"{feature_type_column_name}_creation_time_seconds" in results_df.columns:
        print("time and amount of features already stored. -> Done")
        return

    # get all dataset folders
    dataset_folders = get_sub_folders(datasets_folder)

    # key: dataset_id int
    # value: n_features created int
    n_features_created_dict = {}

    # key: dataset_id int
    # value: feature creation time needed in seconds float
    feature_creation_time_needed_dict = {}

    # make features for each dataset
    for dataset_folder in tqdm(dataset_folders):
        # print header
        print_function_header(f"create {feature_type} features")

        print("---")
        print(f"current folder: {dataset_folder}")
        print("---")

        # make file paths
        X_train_file_path = dataset_folder.joinpath(train_filename)
        X_test_file_path = dataset_folder.joinpath(test_filename)

        # get X and y train and test splitted
        X_train, X_test, y_train, y_test = get_X_train_X_test_y_train_y_test(
            dataset_folder=dataset_folder,
            random_state=random_state,
            X_file_name=X_file_name,
            y_file_name=y_file_name,
        )

        # feedback about train data
        print(f"\nX_train shape: {X_train.shape}")
        print(f"target classes: \n{y_train.value_counts()}\n")

        # start timer
        start_time = time()

        if feature_type == "pca":
            df_train, df_test = _create_pca_features(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                params=transformer_params,
                prefix=prefix,
                mode=pca_mode,
                random_state=random_state,
                early_stopping=10,
                max_n_components_to_create=100,
            )

        elif feature_type == "umap":
            df_train, df_test = _create_umap_features(
                X_train=X_train,
                X_test=X_test,
                params=transformer_params,
                prefix=prefix,
                random_state=random_state,
                range_n_components=umap_range_n_components,
                y_train=y_train
            )

        elif feature_type == "kmeans":
            df_train, df_test = _create_kmeans_features(
                X_train=X_train,
                X_test=X_test,
                params=transformer_params,
                prefix=prefix,
                n_cluster_range=kmeans_n_cluster_range,
                random_state=random_state,
            )

        elif feature_type == "lda":
            df_train, df_test = _create_lda_features(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                params=transformer_params,
                prefix=prefix,
                random_state=random_state,
            )

        else:
            raise NotImplemented(f"{feature_type} is not implemented.")

        # end time
        end_time = time()

        # get amount of new features created
        n_new_features = len(df_train.columns)

        # fill the dicts
        n_features_created_dict[int(dataset_folder.name)] = n_new_features
        feature_creation_time_needed_dict[int(dataset_folder.name)] = end_time - start_time

        # give feedback about new featuers and time needed for fit/transform
        print(f"Created {n_new_features} in {end_time - start_time} seconds")

        # check for NaN
        if df_train.isnull().values.any():
            raise ValueError("train dataframe features contain Nan Values")
        if df_test.isnull().values.any():
            raise ValueError("test dataframe features contain Nan Values")

        # store new features
        df_train.to_feather(X_train_file_path)
        df_test.to_feather(X_test_file_path)

    # sort the dicts by keys to get the same order as the dataframe we want to concat with
    n_features_created_dict = dict(sorted((n_features_created_dict.items())))
    feature_creation_time_needed_dict = dict(sorted((feature_creation_time_needed_dict.items())))

    # add 2 new columns to the results dataframe one for the number of new features and one for the fit-transform time
    # of the feature generation
    results_df[f"{feature_type_column_name}_n_features_created"] = n_features_created_dict.values()
    results_df[f"{feature_type_column_name}_creation_time_seconds"] = feature_creation_time_needed_dict.values()

    # store results dataframe
    results_df.to_feather(path_results_file)

