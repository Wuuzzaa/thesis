import warnings

from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.util import get_sub_folders


def _boruta_selection(X, y, random_state):
    """
    Function to filter features by using borutaPy.
    https://github.com/scikit-learn-contrib/boruta_py
    based on the idea of this paper:
    Kursa M., Rudnicki W., "Feature Selection with the Boruta Package" Journal of Statistical Software, Vol. 36, Issue 11, Sep 2010

    :param X: dataframe with features
    :param y: pandas.Series. with target variable
    :param random_state: int
    :return: Dataframe X filtered
    """
    # convert the dataframe X and the series y to numpy arrays for boruta
    X_array = X.values
    y_array = y.values

    # Boruta feature selection
    feat_selector = BorutaPy(
        estimator=RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=random_state),
        n_estimators='auto',
        verbose=0,
        random_state=random_state,
        max_iter=80,
    )

    # find all relevant features
    feat_selector.fit(X_array, y_array)

    # Filter the features based on the results of boruta
    # boolean indexing taken from https://stackoverflow.com/a/57090806
    X = X.loc[:, feat_selector.support_]

    return X


def _rfecv_selection(X, y, random_state, max_features=None):
    if max_features is not None and len(X.columns) > max_features:
        warnings.warn("Dataset has a huge amount of features. Use fast preselection")

        estimator = RandomForestClassifier(
                n_estimators=2000,
                max_depth=6,
                n_jobs=-1,
                random_state=random_state,
            )

        selector = SelectFromModel(estimator=estimator, threshold=-np.inf, max_features=max_features)

        selector.fit(X, y)
        selected_features = list(selector.get_feature_names_out())

        X = X[selected_features]

    # ensure we got at least two features left for filtering
    if len(X.columns) < 2:
        warnings.warn("RFECV just had 1 Feature to select on")
        return X

    # select the best features recursive with cross validation
    selector = RFECV(
        estimator=RandomForestClassifier(random_state=random_state, n_jobs=-1),
        step=0.01,
        cv=10,
        n_jobs=-1,
        verbose=0,
        min_features_to_select=10  # cause we use pca with 2 components, so we need at least 2 features.
    )

    selector.fit(X, y)
    selected_features = list(selector.get_feature_names_out())

    return X[selected_features]


def _top_boruta_rfecv_selection(X: pd.DataFrame, y: pd.Series, random_state: int, max_features: int, sample_size: int):
    """
    Feature selection with multiple steps. First boruta is used. After this Recursive Feature Elimination is used
    against the prefiltered features. When there are too many features (max_features) left over by boruta.
    A big random forest (many trees) is used to prefilter before RFECV.

    :param X: Data
    :param y: Target
    :param random_state:
    :param max_features: None is possible for no max features to select
    :param sample_size: int with the sample size to use for feature selection. None use all rows of the data
    :return: Dataframe with selected features
    """
    # sample
    if sample_size is not None and len(X) > sample_size:
        print(f"use sample size of {sample_size}")
        X_select, _, y_select, _ = train_test_split(X, y, train_size=sample_size, random_state=random_state)

    else:
        X_select = X.copy()
        y_select = y.copy()

    # first boruta
    print(f"run boruta start with {len(X.columns)} features")
    X_trans = _boruta_selection(X_select, y_select, random_state=random_state)

    boruta_failed_one_feature_only_left = False

    # pca needs at least 2 features so make sure we get at least 2 otherwise let it crash
    if len(X_trans.columns) < 2:
        warnings.warn(f"boruta filtered shit. Just one feature left. Try recursive feature elimination")
        X_trans = _rfecv_selection(X_select, y_select, random_state=random_state, max_features=max_features)
        boruta_failed_one_feature_only_left = True

    # To avoid this return all featuers when there a just a few left
    if len(X_trans.columns) > 10 and not boruta_failed_one_feature_only_left:
        # then recursive feature elimination
        print(f"run recursive feature elimination start with {len(X_trans.columns)} features")
        X_trans = _rfecv_selection(X_trans, y_select, random_state=random_state, max_features=max_features)

    else:
        print(f"Only {len(X_trans.columns)} features left. Do not use recursive feature elimination.")

    selected_features = X_trans.columns

    return X[selected_features]


def feature_selection(
        random_state: int,
        path_datasets_folder: Path,
        path_results_file: Path,
        X_filtered_file_name: str,
        X_clean_file_name: str,
        y_file_name: str,
        max_features: int,
        sample_size: int
):

    print("")
    print("#"*80)
    print("feature selection".upper())
    print("#" * 80)
    print("")

    # dict: key dataset id as int, value int with the amount of selected features
    dict_n_features_filtered = {}

    dataset_folders = get_sub_folders(path_datasets_folder)

    # select features for each dataset
    dataset_folder: Path
    for dataset_folder in tqdm(dataset_folders):
        print(f"current folder: {dataset_folder}")

        # path for X_filtered file
        X_filtered_file_path = dataset_folder.joinpath(X_filtered_file_name)

        # check if already done
        if X_filtered_file_path.exists():
            warnings.warn("Already done skip selection. Load n features filtered for the results dataframe column")

            n_features_filtered = len(pd.read_feather(X_filtered_file_path).columns)

            # add amount of selected features
            dict_n_features_filtered[int(dataset_folder.name)] = n_features_filtered
            continue

        # load X, y
        X = pd.read_feather(dataset_folder.joinpath(X_clean_file_name))
        y = pd.read_feather(dataset_folder.joinpath(y_file_name)).squeeze()

        # run feature selection
        X_trans: pd.DataFrame = _top_boruta_rfecv_selection(X, y, random_state, max_features, sample_size)

        # feedback
        print()
        print("---")
        print(f"{len(X_trans.columns)} features are selected from {len(X.columns)}")
        print("---")
        print()

        # add amount of selected features
        dict_n_features_filtered[int(dataset_folder.name)] = len(X_trans.columns)

        # store the filtered features
        X_trans.to_feather(X_filtered_file_path)

    # sort dict by keys (int values not str)
    dict_n_features_filtered = dict(sorted((dict_n_features_filtered.items())))

    # add n_features_filtered column to results dataframe
    df_results = pd.read_feather(path_results_file)
    df_results["n_features_filtered"] = dict_n_features_filtered.values()
    df_results.to_feather(path_results_file)


def feature_selection_rfecv(X_train, X_test, y_train, random_state, sample_size=None):
    # sample
    if sample_size is not None and len(X_train) > sample_size:
        print(f"use sample size of {sample_size}")
        X_select, _, y_select, _ = train_test_split(X_train, y_train, train_size=sample_size, random_state=random_state)

    else:
        X_select = X_train.copy()
        y_select = y_train.copy()

    # select the best features recursive with cross validation
    selector = RFECV(
        estimator=RandomForestClassifier(random_state=random_state, n_jobs=-1, max_depth=12),
        step=1,
        cv=5,
        n_jobs=-1,
        verbose=0,
        #min_features_to_select=10
    )

    selector.fit(X_select, y_select)
    selected_features = list(selector.get_feature_names_out())

    return X_train[selected_features], X_test[selected_features]
