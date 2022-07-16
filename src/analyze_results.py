from pathlib import Path
import pandas as pd


def add_compare_scores_columns(results_file_path: Path):
    # read file into dataframe
    df = pd.read_feather(results_file_path)

    # compare pca with baseline
    df["pca_clean_train_score > baseline_train_score"]  = df["pca_clean_train_cv_score"] > df["baseline_train_cv_score"]
    df["pca_clean_test_score > baseline_test_score"]    = df["pca_clean_test_score"] > df["baseline_test_score"]

    # compare pca mle with baseline
    df["pca_mle_clean_train_score > baseline_train_score"] = df["pca_mle_clean_train_cv_score"] > df["baseline_train_cv_score"]
    df["pca_mle_clean_test_score > baseline_test_score"] = df["pca_mle_clean_test_score"] > df["baseline_test_score"]

    # compare pca with pca mle
    df["pca_clean_train_score > pca_mle_clean_train_score"] = df["pca_clean_train_cv_score"] > df["pca_mle_clean_train_cv_score"]
    df["pca_clean_test_score > pca_mle_clean_test_score"] = df["pca_clean_test_score"] > df["pca_mle_clean_test_score"]

    # compare pca and pca mle with baseline
    df["pca_clean and pca_clean_mle > baseline train score"] = df["pca_clean_train_score > baseline_train_score"] & df["pca_mle_clean_train_score > baseline_train_score"]
    df["pca_clean and pca_clean_mle > baseline test score"] = df["pca_clean_test_score > baseline_test_score"] & df["pca_mle_clean_test_score > baseline_test_score"]

    # compare kpca with baseline
    df["kpca_clean_train_score > baseline_train_score"]  = df["kpca_clean_train_cv_score"] > df["baseline_train_cv_score"]
    df["kpca_clean_test_score > baseline_test_score"]    = df["kpca_clean_test_score"] > df["baseline_test_score"]

    # store again
    df.to_feather(results_file_path)


def print_info_pca_performance_overview(results_file_path: Path):
    # load results dataframe
    df = pd.read_feather(results_file_path)

    # do some statistics
    n_datasets = len(df)

    # pca
    n_pca_improved_datasets = sum(df['pca_clean_test_score > baseline_test_score'])
    n_pca_mle_improved_datasets = sum(df['pca_mle_clean_test_score > baseline_test_score'])
    pca_improved_dataset_percent = n_pca_improved_datasets / n_datasets
    pca_mle_improved_datasets_percent = n_pca_mle_improved_datasets / n_datasets

    # kpca
    n_kpca_improved_datasets = sum(df['kpca_clean_test_score > baseline_test_score'])
    kpca_improved_dataset_percent = n_kpca_improved_datasets / n_datasets

    # print it out
    print()
    print("#"*80)
    print("pca performance overview".upper())
    print("#" * 80)
    print()
    print(f"Amount of datasets testet: {n_datasets}")
    print("")
    print("Statistics on test data".upper())
    print("")
    print("PCA:")
    print(f"pca on clean data improved the performance on {n_pca_improved_datasets} datasets = {pca_improved_dataset_percent}%")
    print(f"pca on clean data with mle improved the performance on {n_pca_mle_improved_datasets} datasets = {pca_mle_improved_datasets_percent}%")
    print("")
    print("KPCA:")
    print(f"kpca on clean data improved the performance on {n_kpca_improved_datasets} datasets = {kpca_improved_dataset_percent}%")
