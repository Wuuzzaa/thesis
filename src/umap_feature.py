import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from umap import UMAP


def _create_umap_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    params: dict,
    prefix: str,
    random_state: int,
    y_train: pd.Series,
    range_n_components: range = range(1, 6),
):
    # set params
    transformer = UMAP(**params)

    # make a random forest instance
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1, max_depth=10)

    # cv_scores_dict
    # key n_components: int
    # value cv_score: float
    cv_scores_dict = {}

    # check if we use a subset when X is huge
    X_train_sample = X_train.copy()
    y_train_sample = y_train.copy()

    # ignore the train size seems to run slower with fewer samples wtf see test in umap main

    # max_train_size = 10_000
    # if len(X_train) > max_train_size:
    #     print(f"X is too huge use a subset for computational speed. {max_train_size} instead of {len(X_train)} samples are used.")
    #     X_train_sample["y"] = y_train_sample
    #     X_train_sample = X_train_sample.sample(n=max_train_size, random_state=random_state)
    #
    #     # reset index
    #     X_train_sample = X_train_sample.reset_index(drop=True)
    #
    #     y_train_sample = X_train_sample["y"]
    #     X_train_sample = X_train_sample.drop(columns="y")

    # just use a part of the data to avoid overfitting
    size = int(len(X_train_sample) / 3)

    print(f"just use one third of the data {size} samples to avoid overfitting")
    X_train_sample = X_train_sample.head(size)
    y_train_sample = y_train_sample.head(size)

    # just for feedback in the train time print baseline score
    print("\n---")
    print("Cross validation score without umap features:")
    print(cross_val_score(rf, X_train, y_train, cv=5, n_jobs=-1).mean())
    print("---\n")

    # search best n_components with brute force and cross validation score
    for n_components in range_n_components:
        print(f"calculate cross validation score for n_components: {n_components}")
        transformer.set_params(**{"n_components": n_components})

        transformer.fit(X_train_sample, y_train_sample)
        X_train_trans = pd.DataFrame(transformer.transform(X_train)).add_prefix(prefix)

        # concat baseline features with umap features
        X_train_trans_baseline = pd.concat([X_train, X_train_trans], axis="columns")

        # calc cross validation score using umap and baseline features
        cv_score = cross_val_score(rf, X_train_trans_baseline, y_train, cv=5, n_jobs=-1).mean()

        cv_scores_dict[n_components] = cv_score
        print(f"Cross validation score: {cv_score}\n")

    # get the optimal n_components
    optimal_n_components_by_cv_score = max(cv_scores_dict, key=cv_scores_dict.get)
    print(f"optimal n_components: {optimal_n_components_by_cv_score}")

    # set the optimal n_components
    transformer.set_params(**{"n_components": optimal_n_components_by_cv_score})

    print("umap fit")
    # fit with optimal k
    # UMAP can use the target info (y_train) too. Seems to overfit too hard so do not use it.
    # When the whole train data is used it overfits just use less data?
    transformer.fit(X_train_sample)

    # X_train umap features dataframe
    print("umap transform train")
    df_umap_train = pd.DataFrame(transformer.transform(X_train)).add_prefix(prefix)

    # X_test pca features dataframe
    print("umap transform test")
    df_umap_test = pd.DataFrame(transformer.transform(X_test)).add_prefix(prefix)

    return df_umap_train, df_umap_test

    #todo POSSIBLE BUG BUT NOT ON THOSE DATASETS SO LET IT RUN ;-)

    # Traceback (most recent call last):
    #   File "C:\Program Files\JetBrains\PyCharm 2022.1.3\plugins\python\helpers\pydev\pydevd.py", line 1491, in _exec
    #     pydev_imports.execfile(file, globals, locals)  # execute the script
    #   File "C:\Program Files\JetBrains\PyCharm 2022.1.3\plugins\python\helpers\pydev\_pydev_imps\_pydev_execfile.py", line 18, in execfile
    #     exec(compile(contents+"\n", file, 'exec'), glob, loc)
    #   File "C:/Users/jonas/PycharmProjects/thesis/src/testbert.py", line 37, in <module>
    #     X_test_trans = transformer.transform(X_test)
    #   File "C:\Users\jonas\PycharmProjects\thesis\venv\lib\site-packages\umap\umap_.py", line 2896, in transform
    #     indices, dists = self._knn_search_index.query(
    #   File "C:\Users\jonas\PycharmProjects\thesis\venv\lib\site-packages\pynndescent\pynndescent_.py", line 1627, in query
    #     self._init_search_graph()
    #   File "C:\Users\jonas\PycharmProjects\thesis\venv\lib\site-packages\pynndescent\pynndescent_.py", line 981, in _init_search_graph
    #     self._search_forest = [
    #   File "C:\Users\jonas\PycharmProjects\thesis\venv\lib\site-packages\pynndescent\pynndescent_.py", line 982, in <listcomp>
    #     convert_tree_format(tree, self._raw_data.shape[0])
    #   File "C:\Users\jonas\PycharmProjects\thesis\venv\lib\site-packages\pynndescent\rp_trees.py", line 1158, in convert_tree_format
    #     hyperplane_dim = dense_hyperplane_dim(tree.hyperplanes)
    #   File "C:\Users\jonas\PycharmProjects\thesis\venv\lib\site-packages\pynndescent\rp_trees.py", line 1140, in dense_hyperplane_dim
    #     raise ValueError("No hyperplanes of adequate size were found!")
    # ValueError: No hyperplanes of adequate size were found!

    return df_train, df_test


if __name__ == "__main__":
    from calc_scores import get_X_train_X_test_y_train_y_test
    from pathlib import Path
    from constants import *

    # load some data
    X_train, X_test, y_train, y_test = get_X_train_X_test_y_train_y_test(
        dataset_folder=Path(r"C:\Users\jonas\PycharmProjects\thesis\data\datasets\40927"),
        random_state=42,
        X_file_name=X_CLEAN_FILE_NAME,
        y_file_name=Y_FILE_NAME,
    )

    # make umap instance
    UMAP_PARAMS = {
        # for clustering https://umap-learn.readthedocs.io/en/latest/clustering.html
        "n_neighbors": 30,  # default 15. Should be increased to 30
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbose": False,
        "min_dist": 0,
        "n_components": 2,
    }

    transformer = UMAP(**UMAP_PARAMS)

    # fit
    n_head = 1000
    transformer.fit(X_train.head(n_head), y_train.head(n_head))

    # transform
    X_train_trans = transformer.transform(X_train)



