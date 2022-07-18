from pathlib import Path
import pandas as pd


def add_compare_scores_columns(results_file_path: Path):
    # read file into dataframe
    df = pd.read_feather(results_file_path)

    # compare pca with baseline
    df["pca_clean_train_score > baseline_train_score"]  = df["pca_clean_train_cv_score"] > df["baseline_train_cv_score"]
    df["pca_clean_test_score > baseline_test_score"]    = df["pca_clean_test_score"] > df["baseline_test_score"]

    # compare kpca with baseline
    df["kpca_clean_train_score > baseline_train_score"]  = df["kpca_clean_train_cv_score"] > df["baseline_train_cv_score"]
    df["kpca_clean_test_score > baseline_test_score"]    = df["kpca_clean_test_score"] > df["baseline_test_score"]

    # kpca and pca > baseline
    df["pca_clean_test_score & kpca_clean_test_score > baseline_test_score"] = df["pca_clean_test_score > baseline_test_score"] & df["kpca_clean_test_score > baseline_test_score"]
    df["pca_clean_train_score & kpca_clean_train_score > baseline_train_score"] = df["pca_clean_train_score > baseline_train_score"] & df["kpca_clean_train_score > baseline_train_score"]
    
    # kpca and pca > baseline on train and test at the same time
    df["pca_kpca_clean_train_and_test_score > baseline_train_and_test_score"] = df["pca_clean_test_score & kpca_clean_test_score > baseline_test_score"] & df["pca_clean_train_score & kpca_clean_train_score > baseline_train_score"]

    # pca and kpca MERGED
    df["pca_kpca_merged_clean_train_score > baseline_train_score"] = df["pca_and_kpca_clean_train_cv_score"] > df["baseline_train_cv_score"]
    df["pca_kpca_merged_clean_test_score > baseline_test_score"] = df["pca_and_kpca_clean_test_score"] > df["baseline_test_score"]


    # store again
    df.to_feather(results_file_path)


def print_info_pca_performance_overview(results_file_path: Path):
    # load results dataframe
    df = pd.read_feather(results_file_path)

    # do some statistics
    n_datasets = len(df)

    # pca test data
    n_pca_improved_datasets_test = sum(df['pca_clean_test_score > baseline_test_score'])
    pca_improved_dataset_percent_test = round(n_pca_improved_datasets_test / n_datasets * 100, 2)

    # pca train data
    n_pca_improved_datasets_train = sum(df['pca_clean_train_score > baseline_train_score'])
    pca_improved_dataset_percent_train = round(n_pca_improved_datasets_train / n_datasets * 100, 2)

    # kpca test data
    n_kpca_improved_datasets_test = sum(df['kpca_clean_test_score > baseline_test_score'])
    kpca_improved_dataset_percent_test = round(n_kpca_improved_datasets_test / n_datasets * 100, 2)

    # kpca train data
    n_kpca_improved_datasets_train = sum(df['kpca_clean_train_score > baseline_train_score'])
    kpca_improved_dataset_percent_train = round(n_kpca_improved_datasets_train / n_datasets * 100, 2)

    # pca and kpca baseline (not merged, counted is when pca and kpca improved the score compared to the baseline)
    n_pca_and_kpca_improved_datasets = sum(df["pca_clean_test_score & kpca_clean_test_score > baseline_test_score"])
    pca_and_kpca_improved_datasets_percent = round(n_pca_and_kpca_improved_datasets / n_datasets * 100, 2)

    # pca and kpca on train and test improved baseline
    n_pca_and_kpca_improved_datasets_on_train_and_test = sum(df["pca_kpca_clean_train_and_test_score > baseline_train_and_test_score"])
    n_pca_and_kpca_improved_datasets_on_train_and_test_percent = round(n_pca_and_kpca_improved_datasets_on_train_and_test /n_datasets * 100, 2)

    # pca and kpca merged test data
    n_pca_kpca_merged_improved_datasets_test = sum(df["pca_kpca_merged_clean_test_score > baseline_test_score"])
    pca_kpca_merged_improved_dataset_percent_test = round(n_pca_kpca_merged_improved_datasets_test / n_datasets * 100, 2)

    # pca and kpca merged train data
    n_pca_kpca_merged_improved_datasets_train = sum(df["pca_kpca_merged_clean_train_score > baseline_train_score"])
    pca_kpca_merged_improved_dataset_percent_train = round(n_pca_kpca_merged_improved_datasets_train / n_datasets * 100, 2)

    # print it out
    print()
    print("#"*80)
    print("pca performance overview".upper())
    print("#" * 80)
    print()
    print(f"Amount of datasets tested: {n_datasets}")
    print("")
    print("#"*80)

    # TEST DATA

    print("Statistics on test data".upper())
    print("#" * 80)
    print("")
    print("PCA:")
    print(f"pca on clean data improved the performance on {n_pca_improved_datasets_test} datasets = {pca_improved_dataset_percent_test}%")
    print("")
    print("KPCA:")
    print(f"kpca on clean data improved the performance on {n_kpca_improved_datasets_test} datasets = {kpca_improved_dataset_percent_test}%")
    print("")
    print("PCA AND KPCA MERGED")
    print(f"pca and kpca merged on clean data improved the performance on {n_pca_kpca_merged_improved_datasets_test} datasets = {pca_kpca_merged_improved_dataset_percent_test}%")

    # TRAIN DATA

    print("")
    print("#"*80)
    print("Statistics on train data".upper())
    print("#"*80)
    print("")
    print("PCA:")
    print(f"pca on clean data improved the cross validation performance on {n_pca_improved_datasets_train} datasets = {pca_improved_dataset_percent_train}%")
    print("")
    print("KPCA:")
    print(f"kpca on clean data improved the cross validation performance on {n_kpca_improved_datasets_train} datasets = {kpca_improved_dataset_percent_train}%")
    print("")
    print("PCA AND KPCA MERGED")
    print(f"pca and kpca merged on clean data improved the performance on {n_pca_kpca_merged_improved_datasets_train} datasets = {pca_kpca_merged_improved_dataset_percent_train}%")

    # OTHER STUFF

    print("")
    print("#" * 80)
    print("PCA AND KPCA AT THE SAME TIME EACH ON ITS OWN AGAINST  (NOT MERGED)")
    print("#" * 80)
    print(f"pca and kpca on clean data improved the performance on {n_pca_and_kpca_improved_datasets} datasets = {pca_and_kpca_improved_datasets_percent}%")
    print("")
    print("#" * 80)
    print("TRAIN AND TEST OF PCA AND KPCA (NOT MERGED)")
    print("#" * 80)
    print(f"pca and kpca on clean data improved the performance on {n_pca_and_kpca_improved_datasets_on_train_and_test} datasets = {n_pca_and_kpca_improved_datasets_on_train_and_test_percent}%")