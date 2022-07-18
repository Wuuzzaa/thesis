from pathlib import Path
import openml

from load_and_clean_suite_datasets import load_and_clean_suite_datasets
from extract_datasets_info import extract_datasets_info, extract_amount_ohe_features
from calc_scores import calc_scores
from analyze_results import add_compare_scores_columns, print_info_pca_performance_overview

if __name__ == "__main__":
    random_state = 42

    ####################################################################################################################
    # LOAD DATA PREPROCESSING
    ####################################################################################################################

    # load suite
    # https://openml.github.io/openml-python/main/examples/20_basic/simple_suites_tutorial.html#sphx-glr-examples-20-basic-simple-suites-tutorial-py
    suite = openml.study.get_suite(99)

    # first load the datasets from the suit and use ohe etc
    load_and_clean_suite_datasets(suite, random_state)

    # extract infos like amount features, classes etc.
    extract_datasets_info(suite)

    # amount features after one hot encoding
    extract_amount_ohe_features(
        path_datasets_folder=Path("..//data//datasets"),
        path_results_file=Path("..//data//results//results.feather"),
    )

    ####################################################################################################################
    # BASELINE
    ####################################################################################################################

    # calc the train and test scores for the "baseline".
    # "baseline": see calc_scores docu
    calc_scores(
        random_state=random_state,
        path_datasets_folder=Path("..//data//datasets"),
        path_results_file=Path("..//data//results//results.feather"),
        mode="baseline",
    )

    ####################################################################################################################
    # PCA
    ####################################################################################################################

    # calc the train and test scores for the "pca_clean".
    # "pca_clean": see calc_scores docu

    pca_params = {
        "n_components": 3,
        "random_state": random_state
    }

    calc_scores(
        random_state=random_state,
        path_datasets_folder=Path("..//data//datasets"),
        path_results_file=Path("..//data//results//results.feather"),
        mode="pca_clean",
        X_train_pca_file_name="pca_train_clean.feather",
        X_test_pca_file_name="pca_test_clean.feather",
        pca_params=pca_params,
        prefix="pca_",
    )

    # # calc the train and test scores for the "pca_mle_clean".
    # # "pca_mle_clean": see calc_scores docu
    #
    # pca_params = {
    #     "n_components": "mle",
    #     "random_state": random_state
    # }
    #
    # calc_scores(
    #     random_state=random_state,
    #     path_datasets_folder=Path("..//data//datasets"),
    #     path_results_file=Path("..//data//results//results.feather"),
    #     mode="pca_mle_clean",
    #     X_train_pca_file_name="pca_train_mle_clean.feather",
    #     X_test_pca_file_name="pca_test_mle_clean.feather",
    #     pca_params=pca_params,
    #     prefix="pca_mle_",
    # )

    ####################################################################################################################
    # KERNEL PCA
    ####################################################################################################################

    # calc the train and test scores for the "kpca_clean".
    # "kpca_clean": see calc_scores docu

    pca_params = {
        "n_components": 3,
        "random_state": random_state,
        "kernel": "rbf",
        "n_jobs": -1,
        "copy_X": False,
        "eigen_solver": "randomized"  # "auto" did run in a first test. "randomized" is faster and should be used when n_components is low.
    }

    calc_scores(
        random_state=random_state,
        path_datasets_folder=Path("..//data//datasets"),
        path_results_file=Path("..//data//results//results.feather"),
        mode="kpca_clean",
        X_train_pca_file_name="kpca_train_clean.feather",
        X_test_pca_file_name="kpca_test_clean.feather",
        pca_params=pca_params,
        prefix="kpca_",
    )

    ####################################################################################################################
    # PCA AND KERNEL PCA TOGETHER
    ####################################################################################################################

    # calc the train and test scores for the "pca_and_kpca_clean".
    # "pca_and_kpca_clean": see calc_scores docu

    calc_scores(
        random_state=random_state,
        path_datasets_folder=Path("..//data//datasets"),
        path_results_file=Path("..//data//results//results.feather"),
        mode="pca_and_kpca_clean",
    )


    ####################################################################################################################
    # RESULTS STATISTICS
    ####################################################################################################################

    add_compare_scores_columns(results_file_path=Path("..//data//results//results.feather"))
    print_info_pca_performance_overview(results_file_path=Path("..//data//results//results.feather"))




