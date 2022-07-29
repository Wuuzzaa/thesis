import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


def _create_kmeans_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    params: dict,
    prefix: str,
    n_cluster_range: range,
    random_state: int,
):
    """
    See silhouette score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

    :param X_train:
    :param X_test:
    :param params:
    :param prefix:
    :param n_cluster_range:
    :return:
    """

    # set params
    transformer = MiniBatchKMeans(**params)

    # dict key: k value: silhouette score
    silhouette_scores_dict = {}

    # check if we use a subset when X is huge
    X_train_fit = X_train.copy()

    max_train_size = 10_000
    if len(X_train) > max_train_size:
        print(f"X is too huge use a subset for computational speed. {max_train_size} instead of {len(X_train)} samples are used.")
        X_train_fit = X_train.sample(n=max_train_size, random_state=random_state)

    # search best k with the silhouette score
    for k in n_cluster_range:
        print(f"calculate silhouette score for k: {k}")
        transformer.set_params(**{"n_clusters": k})

        transformer.fit(X_train_fit)

        score = silhouette_score(X_train_fit, transformer.labels_, metric='euclidean')
        silhouette_scores_dict[k] = score
        print(f"silhouette score: {score}")

    # optimal k is the k with the highest silhouette score
    k = max(silhouette_scores_dict, key=silhouette_scores_dict.get)

    # fit with optimal k
    print(f"fit with optimal k: {k}")
    transformer.set_params(**{"n_clusters": k})
    transformer.fit(X_train_fit)

    # predict
    print("predict train data")
    df_train = pd.DataFrame(transformer.predict(X_train)).add_prefix(prefix)

    print("predict test data")
    df_test = pd.DataFrame(transformer.predict(X_test)).add_prefix(prefix)

    return df_train, df_test




