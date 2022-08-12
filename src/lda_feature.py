import pandas as pd
import warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def _create_lda_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        params: dict,
        prefix: str,
        random_state: int,
) -> (pd.DataFrame, pd.DataFrame):
    # make a random forest
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    # cross validation score dict
    # key: n_components int
    # value: mean cross validation score float
    cv_scores_dict = {}

    # make a transformer and set the params
    transformer = LinearDiscriminantAnalysis(**params)

    # brute force the n_components for lda by the cv_score of the random forest
    n_classes = len(y_train.value_counts())

    # lda restrictions:
    # n_components cannot be larger than min(n_features, n_classes - 1)
    upper_bound_n_components = min(len(X_train.columns), n_classes)  # not -1 because we run a for loop till -1 anyway

    # not +1 because lda restrictions
    for n_components in range(1, upper_bound_n_components):
        print(f"test {n_components} components")

        # set n_components
        transformer.set_params(**{"n_components": n_components})

        X_temp_train = transformer.fit_transform(X_train.copy(), y_train)

        # use baseline features and lda features for the cross validation score calculation
        X_temp_train_baseline = pd.concat([X_temp_train, X_train], axis="columns")

        cv_score = cross_val_score(estimator=rf, X=X_temp_train_baseline, y=y_train, cv=5, n_jobs=-1).mean()
        print(f"cv_score: {cv_score}")

        cv_scores_dict[n_components] = cv_score

    # best found n_components is the one with the highest cv_score
    best_n_components = max(cv_scores_dict, key=cv_scores_dict.get)

    # set the best n components
    transformer.set_params(**{"n_components": best_n_components})

    # fit transform
    transformer.fit(X_train, y_train)

    df_train = pd.DataFrame(transformer.transform(X_train)).add_prefix(prefix)
    df_test = pd.DataFrame(transformer.transform(X_test)).add_prefix(prefix)

    return df_train, df_test
