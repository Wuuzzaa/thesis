import warnings
from pathlib import Path
import openml
import pandas as pd
from tqdm import tqdm


def extract_datasets_info(suite):
    """
    Function to extract the following infos from each task/dataset:
        -dataset_id
        -dataset_name
        -task_id
        -n_classes
        -n_features
        -n_samples

    The infos are stored as a dataframe

    :param suite: openml suit to extract the infos from.
    :return: None
    """
    print("")
    print("#"*80)
    print("extract datasets info".upper())
    print("#" * 80)
    print("")

    path = Path(f"..//data//results")
    path_results_file = path.joinpath("results.feather")

    # check if already done
    if path_results_file.exists():
        print("already done... return")
        return

    # lists to store data for dataframe
    dataset_ids = []
    task_ids = []
    n_classes = []
    n_features = []
    name_datasets = []
    n_samples = []

    # get all tasks of the study
    tasks = suite.tasks

    for task_id in tasks:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        # get X as Dataframe and y as Series
        target = dataset.default_target_attribute
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=target)

        dataset_ids.append(dataset.id)
        task_ids.append(task_id)
        n_classes.append(len(task.class_labels))
        n_features.append(len(dataset.features) - 1)  # -1 because the target is in the list of features
        name_datasets.append(dataset.name)
        n_samples.append(len(X))

    data = {
        "dataset_id": dataset_ids,
        "dataset_name": name_datasets,
        "task_id": task_ids,
        "n_classes": n_classes,
        "n_features": n_features,
        "n_samples": n_samples
    }

    df = pd.DataFrame(data).\
        sort_values(by=["dataset_id"]).\
        reset_index(drop=True)

    # create folder for dataset
    path.mkdir(parents=True, exist_ok=True)

    # store dataframe
    df.to_feather(path=path_results_file)


def extract_amount_ohe_features(path_datasets_folder, path_results_file):
    print("")
    print("#"*80)
    print("extract amount ohe features".upper())
    print("#" * 80)
    print("")

    ohe_column_name = "n_features_ohe"

    # load the results dataframes columns
    results_df_columns = pd.read_feather(path_results_file).columns.tolist()

    # # when the ohe column is already in the dataframe we can skip
    # if ohe_column_name in results_df_columns:
    #     warnings.warn(f"{ohe_column_name} is already in the results dataframe. Done")
    #     return

    # make some lists
    n_features_ohe_dict = {}
    dataset_folders = []

    # get the dataset folders in the data folder
    for path in path_datasets_folder.iterdir():
        if path.is_dir():
            dataset_folders.append(path)

    for dataset_folder in tqdm(dataset_folders):
        print(f"current folder: {dataset_folder}")

        # get X columns
        path_X = dataset_folder.joinpath("X_clean.feather")
        X_columns = pd.read_feather(path_X).columns.tolist()
        n_features_ohe_dict[int(dataset_folder.name)] = len(X_columns)

    # load the results dataframe
    results_df = pd.read_feather(path_results_file)

    # sort the n_features_ohe_dict by the keys
    n_features_ohe_dict = dict(sorted((n_features_ohe_dict.items())))

    # add the new column
    results_df[ohe_column_name] = pd.Series(data=n_features_ohe_dict.values())

    # store results with ohe features info
    results_df.to_feather(path_results_file)
