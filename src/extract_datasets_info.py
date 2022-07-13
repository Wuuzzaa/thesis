from pathlib import Path
import openml
import pandas as pd


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
    n_features_ohe = []  # todo get data from X feather files
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