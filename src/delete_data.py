from pathlib import Path
import pandas as pd
from constants import *


def delete_files(root_folder: Path, filename: str):
    """
    All child folders of the given root folder are traversed in search for files with the given filename to delete


    :param root_folder: Path of the rootfolder.
    :param filename: filename to delete
    :return:
    """
    to_delete = []

    # travers root folder
    for path in root_folder.rglob("*"):
        if path.is_file():
            if path.name == filename:
                to_delete.append(path)

    # delete
    for path in to_delete:
        print(f"delete file: {path}")
        path.unlink()


def delete_columns_from_results_file(columns: list[str]):
    df_results: pd.DataFrame

    # load results
    df_results = pd.read_feather(RESULTS_FILE_PATH)

    # drop
    df_results = df_results.drop(columns=columns, errors="ignore")

    # store again
    df_results.to_feather(RESULTS_FILE_PATH)


if __name__ == "__main__":
    from constants import *
    root_folder = DATA_FOLDER_PATH

    # set a filename
    filenames = [
        # #X and y
        # X_CLEAN_FILE_NAME,
        # X_FILTERED_FILE_NAME,
        # Y_FILE_NAME,

        # # pca files
        # X_TRAIN_CLEAN_PCA_FILE_NAME,
        # X_TEST_CLEAN_PCA_FILE_NAME,
        #
        # # kpca files
        # X_TRAIN_CLEAN_KPCA_FILE_NAME,
        # X_TEST_CLEAN_KPCA_FILE_NAME,

        # # umap files
        # X_TRAIN_CLEAN_UMAP_FILE_NAME,
        # X_TEST_CLEAN_UMAP_FILE_NAME,

        ## kmeans files
        # X_TRAIN_CLEAN_KMEANS_FILE_NAME,
        # X_TEST_CLEAN_KMEANS_FILE_NAME,

        # # results file
        # RESULTS_DATAFRAME_FILE_NAME,

        # # feature importance files
        # FEATURE_IMPORTANCE_FILE_NAME,

        # # temp filenames adjust them later
        # "baseline_random_forest.joblib",
        # "kpca_clean_random_forest.joblib",
        # "pca_and_kpca_clean_random_forest.joblib",
        # "pca_clean_random_forest.joblib",
        # "umap_clean_random_forest.joblib",
    ]

    if len(filenames) > 0:
        print("Are you sure to delete all files with the following names?")

        [print(filename) for filename in filenames]

        saftycheck = input("\nWrite 'delete' to remove all of them\n")

        if saftycheck == "delete":
            # traverse the root folder and subfolders and delete all matches
            for filename in filenames:
                print(f"delete all files with name: {filename}")
                delete_files(root_folder, filename)

        else:
            print(f"input was {saftycheck}. No files deleted")


    ####################################################################################################################
    # delete columns if needed
    ####################################################################################################################

    columns_to_delete = [
        #     #"dataset_id",
        #     #"dataset_name",
        #     #"task_id",
        #     #"n_classes",
        #     #"n_features",
        #     #"n_samples",
        #     #"n_features_ohe",
        #     #"n_features_filtered",
        #     #"baseline_train_cv_score",
        #     #"baseline_test_score",
        #     #"pca_clean_train_cv_score",
        #     #"pca_clean_test_score",
        #     #"kpca_clean_train_cv_score",
        #     #"kpca_clean_test_score",
        #     #"pca_clean_pca_features_importance_mean_factor",
        #     #"kpca_clean_pca_features_importance_mean_factor",
        #     #"pca_clean_train_score > baseline_train_score",
        #     #"pca_clean_test_score > baseline_test_score",
        #     #"kpca_clean_train_score > baseline_train_score",
        #     #"kpca_clean_test_score > baseline_test_score",
        #     #"pca_clean_test_score & kpca_clean_test_score > baseline_test_score",
        #     #"pca_clean_train_score & kpca_clean_train_score > baseline_train_score",
        #     #"pca_kpca_clean_train_and_test_score > baseline_train_and_test_score",
        #     #"pca_kpca_merged_clean_train_score > baseline_train_score",
        #     #"pca_kpca_merged_clean_test_score > baseline_test_score",
        #     #"pca_clean_test_score_change_to_baseline",
        #     #"kpca_clean_test_score_change_to_baseline",
        #     #"model_hyperparameter_baseline_random_forest",
        #     #"model_hyperparameter_pca_clean_random_forest",
        #     #"model_hyperparameter_kpca_clean_random_forest",
        #     #"umap_clean_train_cv_score",
        #     #"umap_clean_test_score",
        #     "umap_clean_train_score > baseline_train_score",
        #     "umap_clean_test_score > baseline_test_score",
        #     "umap_clean_test_score_change_to_baseline",
        #     "model_hyperparameter_umap_clean_random_forest",
        #     "any_feature_type_test_score > baseline_test_score",
        #     #"kmeans_clean_train_cv_score",
        #     #"kmeans_clean_test_score",
        #     "kmeans_clean_train_score > baseline_train_score",
        #     "kmeans_clean_test_score > baseline_test_score",
        #     "kmeans_clean_test_score_change_to_baseline",
        #     "model_hyperparameter_kmeans_clean_random_forest",
    ]

    if len(columns_to_delete) > 0:
        print("Are you sure to delete all columns with the following names?")

        [print(column) for column in columns_to_delete]

        saftycheck = input("\nWrite 'delete' to remove all of them\n")

        if saftycheck == "delete":
            delete_columns_from_results_file(columns_to_delete)

        else:
            print(f"input was {saftycheck}. No columns deleted")
