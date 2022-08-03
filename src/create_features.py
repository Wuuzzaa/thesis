import warnings
from src.calc_scores import get_X_train_X_test_y_train_y_test
from src.kmeans_feature import _create_kmeans_features
from src.lda_feature import _create_lda_features
from src.pca_feature import _create_pca_features
from src.stacking_feature import _create_stacking_features
from src.umap_feature import _create_umap_features
from src.util import print_function_header, get_sub_folders
from pathlib import Path
from tqdm import tqdm


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
        pca_mode: str=None,
        kmeans_n_cluster_range: range = None,
):
    # print header
    print_function_header(f"create {feature_type} features")

    # get all dataset folders
    dataset_folders = get_sub_folders(datasets_folder)

    # make features for each dataset
    for dataset_folder in tqdm(dataset_folders):
        print("---")
        print(f"current folder: {dataset_folder}")
        print("---")

        # make file paths
        X_train_file_path = dataset_folder.joinpath(train_filename)
        X_test_file_path = dataset_folder.joinpath(test_filename)

        # check if file already exists -> load from files the features
        if X_train_file_path.is_file() and X_test_file_path.is_file():
            warnings.warn(f"feature files found  {X_train_file_path} and {X_test_file_path}. Done")
            continue

        # get X and y train and test splitted
        X_train, X_test, y_train, y_test = get_X_train_X_test_y_train_y_test(
            dataset_folder=dataset_folder,
            random_state=random_state,
            X_file_name=X_file_name,
            y_file_name=y_file_name,
        )

        print(f"X_train shape: {X_train.shape}")
        print(f"target classes: \n{y_train.value_counts()}")

        if feature_type == "pca":
            # make the features dataframes for train and test
            df_train, df_test = _create_pca_features(
                X_train=X_train,
                X_test=X_test,
                params=transformer_params,
                prefix=prefix,
                mode=pca_mode,
                random_state=random_state,
            )

        elif feature_type == "umap":
            # make the features dataframes for train and test
            # do not use y_train even when umap is able to do it -> it just overfits
            df_train, df_test = _create_umap_features(
                X_train=X_train,
                X_test=X_test,
                params=transformer_params,
                prefix=prefix,
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
            )

        elif feature_type == "stacking":
            df_train, df_test = _create_stacking_features(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                params=transformer_params,
                prefix=prefix,
            )

        else:
            raise NotImplemented(f"{feature_type} is not implemented.")

        # check for NaN
        if df_train.isnull().values.any():
            raise ValueError("train dataframe features contain Nan Values")
        if df_test.isnull().values.any():
            raise ValueError("test dataframe features contain Nan Values")

        # store new features
        df_train.to_feather(X_train_file_path)
        df_test.to_feather(X_test_file_path)
