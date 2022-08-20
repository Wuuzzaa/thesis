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

        # # lda files
        # X_TRAIN_CLEAN_LDA_FILE_NAME,
        # X_TEST_CLEAN_LDA_FILE_NAME,

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
        #"dataset_id",
        #"dataset_name",
        #"task_id",
        #"n_classes",
        #"n_features",
        #"n_samples",
        #"n_features_ohe",
        #"n_features_filtered",
        #"pca_clean_n_features_created",
        #"pca_clean_creation_time_seconds",
        #"kpca_clean_n_features_created",
        #"kpca_clean_creation_time_seconds",
        #"umap_clean_n_features_created",
        #"umap_clean_creation_time_seconds",
        #"kmeans_clean_n_features_created",
        #"kmeans_clean_creation_time_seconds",
        #"lda_clean_n_features_created",
        #"lda_clean_creation_time_seconds",
        #"pca_filtered_n_features_created",
        #"pca_filtered_creation_time_seconds",
        #"kpca_filtered_n_features_created",
        #"kpca_filtered_creation_time_seconds",
        #"umap_filtered_n_features_created",
        #"umap_filtered_creation_time_seconds",
        #"kmeans_filtered_n_features_created",
        #"kmeans_filtered_creation_time_seconds",
        #"lda_filtered_n_features_created",
        #"lda_filtered_creation_time_seconds",
        #"baseline_filtered_train_cv_score",
        #"baseline_filtered_test_score",
        #"baseline_filtered_train_time_in_seconds",
        #"only_pca_train_cv_score",
        #"only_pca_test_score",
        #"only_pca_train_time_in_seconds",
        #"only_kpca_train_cv_score",
        #"only_kpca_test_score",
        #"only_kpca_train_time_in_seconds",
        #"only_kmeans_train_cv_score",
        #"only_kmeans_test_score",
        #"only_kmeans_train_time_in_seconds",
        #"only_lda_train_cv_score",
        #"only_lda_test_score",
        #"only_lda_train_time_in_seconds",
        #"only_umap_train_cv_score",
        #"only_umap_test_score",
        #"only_umap_train_time_in_seconds",
        #"only_pca_filtered_train_cv_score",
        #"only_pca_filtered_test_score",
        #"only_pca_filtered_train_time_in_seconds",
        #"only_kpca_filtered_train_cv_score",
        #"only_kpca_filtered_test_score",
        #"only_kpca_filtered_train_time_in_seconds",
        #"only_kmeans_filtered_train_cv_score",
        #"only_kmeans_filtered_test_score",
        #"only_kmeans_filtered_train_time_in_seconds",
        #"only_lda_filtered_train_cv_score",
        #"only_lda_filtered_test_score",
        #"only_lda_filtered_train_time_in_seconds",
        #"only_umap_filtered_train_cv_score",
        #"only_umap_filtered_test_score",
        #"only_umap_filtered_train_time_in_seconds",
        #"baseline_filtered_pca_train_cv_score",
        #"baseline_filtered_pca_test_score",
        #"baseline_filtered_pca_train_time_in_seconds",
        #"baseline_filtered_kpca_train_cv_score",
        #"baseline_filtered_kpca_test_score",
        #"baseline_filtered_kpca_train_time_in_seconds",
        #"baseline_filtered_kmeans_train_cv_score",
        #"baseline_filtered_kmeans_test_score",
        #"baseline_filtered_kmeans_train_time_in_seconds",
        #"baseline_filtered_lda_train_cv_score",
        #"baseline_filtered_lda_test_score",
        #"baseline_filtered_lda_train_time_in_seconds",
        #"baseline_filtered_umap_train_cv_score",
        #"baseline_filtered_umap_test_score",
        #"baseline_filtered_umap_train_time_in_seconds",
        #"baseline_filtered_pca_filtered_train_cv_score",
        #"baseline_filtered_pca_filtered_test_score",
        #"baseline_filtered_pca_filtered_train_time_in_seconds",
        #"baseline_filtered_kpca_filtered_train_cv_score",
        #"baseline_filtered_kpca_filtered_test_score",
        #"baseline_filtered_kpca_filtered_train_time_in_seconds",
        #"baseline_filtered_kmeans_filtered_train_cv_score",
        #"baseline_filtered_kmeans_filtered_test_score",
        #"baseline_filtered_kmeans_filtered_train_time_in_seconds",
        #"baseline_filtered_lda_filtered_train_cv_score",
        #"baseline_filtered_lda_filtered_test_score",
        #"baseline_filtered_lda_filtered_train_time_in_seconds",
        #"baseline_filtered_umap_filtered_train_cv_score",
        #"baseline_filtered_umap_filtered_test_score",
        #"baseline_filtered_umap_filtered_train_time_in_seconds",
        #"selected_features_train_cv_score",
        #"selected_features_test_score",
        #"selected_features_train_time_in_seconds",
        #"selected_features_filtered_train_cv_score",
        #"selected_features_filtered_test_score",
        #"selected_features_filtered_train_time_in_seconds",
        #"all_features_train_cv_score",
        #"all_features_test_score",
        #"all_features_train_time_in_seconds",
        #"all_features_filtered_train_cv_score",
        #"all_features_filtered_test_score",
        #"all_features_filtered_train_time_in_seconds",
        #"stacking_baseline_filtered_train_cv_score",
        #"stacking_baseline_filtered_test_score",
        #"stacking_baseline_filtered_train_time_in_seconds",
        #"stacking_all_features_train_cv_score",
        #"stacking_all_features_test_score",
        #"stacking_all_features_train_time_in_seconds",
        #"only_pca_train_score > baseline_filtered_train_score",
        #"only_pca_test_score > baseline_filtered_test_score",
        #"only_pca_test_score_change_to_baseline_filtered",
        #"only_kpca_train_score > baseline_filtered_train_score",
        #"only_kpca_test_score > baseline_filtered_test_score",
        #"only_kpca_test_score_change_to_baseline_filtered",
        #"only_kmeans_train_score > baseline_filtered_train_score",
        #"only_kmeans_test_score > baseline_filtered_test_score",
        #"only_kmeans_test_score_change_to_baseline_filtered",
        #"only_lda_train_score > baseline_filtered_train_score",
        #"only_lda_test_score > baseline_filtered_test_score",
        #"only_lda_test_score_change_to_baseline_filtered",
        #"only_umap_train_score > baseline_filtered_train_score",
        #"only_umap_test_score > baseline_filtered_test_score",
        #"only_umap_test_score_change_to_baseline_filtered",
        #"only_pca_filtered_train_score > baseline_filtered_train_score",
        #"only_pca_filtered_test_score > baseline_filtered_test_score",
        #"only_pca_filtered_test_score_change_to_baseline_filtered",
        #"only_kpca_filtered_train_score > baseline_filtered_train_score",
        #"only_kpca_filtered_test_score > baseline_filtered_test_score",
        #"only_kpca_filtered_test_score_change_to_baseline_filtered",
        #"only_kmeans_filtered_train_score > baseline_filtered_train_score",
        #"only_kmeans_filtered_test_score > baseline_filtered_test_score",
        #"only_kmeans_filtered_test_score_change_to_baseline_filtered",
        #"only_lda_filtered_train_score > baseline_filtered_train_score",
        #"only_lda_filtered_test_score > baseline_filtered_test_score",
        #"only_lda_filtered_test_score_change_to_baseline_filtered",
        #"only_umap_filtered_train_score > baseline_filtered_train_score",
        #"only_umap_filtered_test_score > baseline_filtered_test_score",
        #"only_umap_filtered_test_score_change_to_baseline_filtered",
        #"baseline_filtered_pca_train_score > baseline_filtered_train_score",
        #"baseline_filtered_pca_test_score > baseline_filtered_test_score",
        #"baseline_filtered_pca_test_score_change_to_baseline_filtered",
        #"baseline_filtered_kpca_train_score > baseline_filtered_train_score",
        #"baseline_filtered_kpca_test_score > baseline_filtered_test_score",
        #"baseline_filtered_kpca_test_score_change_to_baseline_filtered",
        #"baseline_filtered_kmeans_train_score > baseline_filtered_train_score",
        #"baseline_filtered_kmeans_test_score > baseline_filtered_test_score",
        #"baseline_filtered_kmeans_test_score_change_to_baseline_filtered",
        #"baseline_filtered_lda_train_score > baseline_filtered_train_score",
        #"baseline_filtered_lda_test_score > baseline_filtered_test_score",
        #"baseline_filtered_lda_test_score_change_to_baseline_filtered",
        #"baseline_filtered_umap_train_score > baseline_filtered_train_score",
        #"baseline_filtered_umap_test_score > baseline_filtered_test_score",
        #"baseline_filtered_umap_test_score_change_to_baseline_filtered",
        #"baseline_filtered_pca_filtered_train_score > baseline_filtered_train_score",
        #"baseline_filtered_pca_filtered_test_score > baseline_filtered_test_score",
        #"baseline_filtered_pca_filtered_test_score_change_to_baseline_filtered",
        #"baseline_filtered_kpca_filtered_train_score > baseline_filtered_train_score",
        #"baseline_filtered_kpca_filtered_test_score > baseline_filtered_test_score",
        #"baseline_filtered_kpca_filtered_test_score_change_to_baseline_filtered",
        #"baseline_filtered_kmeans_filtered_train_score > baseline_filtered_train_score",
        #"baseline_filtered_kmeans_filtered_test_score > baseline_filtered_test_score",
        #"baseline_filtered_kmeans_filtered_test_score_change_to_baseline_filtered",
        #"baseline_filtered_lda_filtered_train_score > baseline_filtered_train_score",
        #"baseline_filtered_lda_filtered_test_score > baseline_filtered_test_score",
        #"baseline_filtered_lda_filtered_test_score_change_to_baseline_filtered",
        #"baseline_filtered_umap_filtered_train_score > baseline_filtered_train_score",
        #"baseline_filtered_umap_filtered_test_score > baseline_filtered_test_score",
        #"baseline_filtered_umap_filtered_test_score_change_to_baseline_filtered",
        #"selected_features_train_score > baseline_filtered_train_score",
        #"selected_features_test_score > baseline_filtered_test_score",
        #"selected_features_test_score_change_to_baseline_filtered",
        #"selected_features_filtered_train_score > baseline_filtered_train_score",
        #"selected_features_filtered_test_score > baseline_filtered_test_score",
        #"selected_features_filtered_test_score_change_to_baseline_filtered",
        #"all_features_train_score > baseline_filtered_train_score",
        #"all_features_test_score > baseline_filtered_test_score",
        #"all_features_test_score_change_to_baseline_filtered",
        #"all_features_filtered_train_score > baseline_filtered_train_score",
        #"all_features_filtered_test_score > baseline_filtered_test_score",
        #"all_features_filtered_test_score_change_to_baseline_filtered",
        #"any_feature_type_clean_test_score > baseline_filtered_test_score",
        #"any_feature_type_clean_filtered_test_score > baseline_filtered_test_score",
        #"model_hyperparameter_baseline_filtered_random_forest",
        #"model_hyperparameter_only_pca_random_forest",
        #"model_hyperparameter_only_kpca_random_forest",
        #"model_hyperparameter_only_kmeans_random_forest",
        #"model_hyperparameter_only_lda_random_forest",
        #"model_hyperparameter_only_umap_random_forest",
        #"model_hyperparameter_only_pca_filtered_random_forest",
        #"model_hyperparameter_only_kpca_filtered_random_forest",
        #"model_hyperparameter_only_kmeans_filtered_random_forest",
        #"model_hyperparameter_only_lda_filtered_random_forest",
        #"model_hyperparameter_only_umap_filtered_random_forest",
        #"model_hyperparameter_baseline_filtered_pca_random_forest",
        #"model_hyperparameter_baseline_filtered_kpca_random_forest",
        #"model_hyperparameter_baseline_filtered_kmeans_random_forest",
        #"model_hyperparameter_baseline_filtered_lda_random_forest",
        #"model_hyperparameter_baseline_filtered_umap_random_forest",
        #"model_hyperparameter_baseline_filtered_pca_filtered_random_forest",
        #"model_hyperparameter_baseline_filtered_kpca_filtered_random_forest",
        #"model_hyperparameter_baseline_filtered_kmeans_filtered_random_forest",
        #"model_hyperparameter_baseline_filtered_lda_filtered_random_forest",
        #"model_hyperparameter_baseline_filtered_umap_filtered_random_forest",
        #"model_hyperparameter_selected_features_random_forest",
        #"model_hyperparameter_selected_features_filtered_random_forest",
        #"model_hyperparameter_all_features_random_forest",
        #"model_hyperparameter_all_features_filtered_random_forest",
        #"model_hyperparameter_stacking_baseline_filtered_stacking",
        #"model_hyperparameter_stacking_all_features_stacking",
    ]

    if len(columns_to_delete) > 0:
        print("Are you sure to delete all columns with the following names?")

        [print(column) for column in columns_to_delete]

        saftycheck = input("\nWrite 'delete' to remove all of them\n")

        if saftycheck == "delete":
            delete_columns_from_results_file(columns_to_delete)

        else:
            print(f"input was {saftycheck}. No columns deleted")
