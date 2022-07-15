import warnings

import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from pathlib import Path
from src.pca_feature import create_pca_features


def calc_scores(
        random_state: int,
        path_datasets_folder: Path,
        path_results_file: Path,
        mode: str,
        X_train_pca_file_name: str = None,
        X_test_pca_file_name: str = None,
        pca_params: dict = None,
        prefix: str = None,
):
    """
    Function to calc cross validation score for the train data and score for the test data. The scores are appended to
    the results file.


    :param random_state:
    :param path_datasets_folder:
    :param path_results_file:
    :param mode:
        "baseline": Runs a random forest on the cleaned data no additional features no feature selection
        "pca_clean": Runs a random forest on the cleaned data with pca additional features no feature selection.
            pca_params = {
            "n_components": 3,
            "random_state": random_state
            }
        "pca_mle_clean": like "pca_clean" mode but n_components is set to "mle"

    :param X_train_pca_file_name: Needed for mode "pca_clean". Just the filename not the path.
    :param X_test_pca_file_name: Needed for mode "pca_clean". Just the filename not the path.
    :param pca_params: Needed for mode "pca_clean". dict with the parameters used for pca.
    For the parameters see: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    :param prefix: Needed for mode "pca_clean". Prefix for the column name in the dataframe for the generated pca features
    :return: None
    """

    print("")
    print("#"*80)
    print("calc scores".upper())
    print(f"mode: {mode}".upper())
    print("#" * 80)
    print("")

    # check needed parameters are not None
    if "pca" in mode:
        needed_pca_parameters = [
            X_train_pca_file_name,
            X_test_pca_file_name,
            pca_params,
            prefix,
        ]

        if None in needed_pca_parameters:
            raise ValueError(f"One or more parameter for pca is None. Give it a value.")

    # check if mode is valid
    modes = [
        "baseline",
        "pca_clean",
        "pca_mle_clean"
    ]
    if mode not in modes:
        raise NotImplemented(f"mode: {mode} is not implemented. Use on of thease modes {modes}")

    # check if already done, by checking if the new to generate columns are already in the results dataframe
    train_cv_score_column_name = f"{mode}_train_cv_score"
    test_score_column_name = f"{mode}_test_score"

    new_columns = [train_cv_score_column_name, test_score_column_name]

    # load the results dataframes columns
    results_df_columns = pd.read_feather(path_results_file).columns.tolist()

    # when all column we would calculate by this mode are already in the results dataframe we can skip it
    if set(new_columns).issubset(results_df_columns):
        warnings.warn(f"All columns for mode: {mode} are already in the results file. Done")
        return

    # make some dicts
    train_cv_scores_dict = {}
    test_scores_dict = {}

    # store the dataset folders
    dataset_folders = []

    # use a random forest as classifier
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1, max_depth=12)

    # get the dataset folders in the data folder
    for path in path_datasets_folder.iterdir():
        if path.is_dir():
            dataset_folders.append(path)

    # score each dataset
    for dataset_folder in tqdm(dataset_folders):
        print(f"current folder: {dataset_folder}")

        if mode == "baseline":
            # get X, y
            path_X = dataset_folder.joinpath("X_clean.feather")
            path_y = dataset_folder.joinpath("y.feather")

            X = pd.read_feather(path_X)
            y = pd.read_feather(path_y)["y"]

            # train test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, train_size=0.75)

        elif mode in ["pca_clean", "pca_mle_clean"]:
            # get X, y
            path_X = dataset_folder.joinpath("X_clean.feather")
            path_y = dataset_folder.joinpath("y.feather")

            X = pd.read_feather(path_X)
            y = pd.read_feather(path_y)["y"]

            # train test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, train_size=0.75)

            df_pca_train, df_pca_test = create_pca_features(
                X_train=X_train,
                X_test=X_test,
                X_train_pca_file=dataset_folder.joinpath(X_train_pca_file_name),
                X_test_pca_file=dataset_folder.joinpath(X_test_pca_file_name),
                pca_params=pca_params,
                prefix=prefix
            )

            # drop index to be able to concat on axis columns
            X_train.reset_index(drop=True, inplace=True)
            X_test.reset_index(drop=True, inplace=True)

            # concat the new features to the old ones
            X_train = pd.concat([X_train, df_pca_train], axis="columns")
            X_test = pd.concat([X_test, df_pca_test], axis="columns")

        # fit model
        rf.fit(X_train, y_train)

        # score model on test data and cv score for train data
        test_scores_dict[int(dataset_folder.name)] = rf.score(X_test, y_test)

        # let n_jobs at 1 here to not run out of RAM. The random forest will use all cores so no problem.
        train_cv_scores_dict[int(dataset_folder.name)] = cross_val_score(rf, X_train, y_train, cv=10, n_jobs=1).mean()

    # sort the dicts by keys to get the same order as the dataframe we want to concat with
    test_scores_dict = dict(sorted((test_scores_dict.items())))
    train_cv_scores_dict = dict(sorted((train_cv_scores_dict.items())))

    # make a dataframe from the score lists
    scores_df = pd.DataFrame(data={
        train_cv_score_column_name: train_cv_scores_dict.values(),
        test_score_column_name: test_scores_dict.values()
    })

    # load the results dataframe
    results_df = pd.read_feather(path_results_file)

    # concat the results dataframe with the scores dataframe
    results_df = pd.concat([results_df, scores_df], axis="columns")

    # store results with scores
    results_df.to_feather(path_results_file)
