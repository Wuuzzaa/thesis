from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV


def boruta_selection(X, y, random_state):
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

    # Boruta feature selectior
    feat_selector = BorutaPy(
        estimator=RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=random_state),
        n_estimators=100,
        verbose=1,
        random_state=random_state,
    )

    # find all relevant features
    feat_selector.fit(X_array, y_array)

    # Filter the features based on the results of boruta
    # boolean indexing taken from https://stackoverflow.com/a/57090806
    X = X.loc[:, feat_selector.support_]

    return X


def rfecv_selection(X, y, random_state):
    selector = RFECV(
        estimator=RandomForestClassifier(max_depth=10, random_state=random_state, n_jobs=-1),
        step=0.01,
        cv=5,
        n_jobs=1,
        verbose=1
    )

    selector.fit(X, y)
    selected_features = list(selector.get_feature_names_out())

    return X[selected_features]


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    from sklearn.model_selection import cross_val_score

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # load X, y
    X = pd.read_feather("..//data//datasets//28//X_clean.feather")
    y = pd.read_feather("..//data//datasets//28//y.feather")["y"]

    # baseline all features
    print(f"X_shape: {X.shape}")
    print(f"cross val score baseline {cross_val_score(estimator=rf, X=X, y=y, cv=10).mean()}")

    #boruta
    X_trans = boruta_selection(X.copy(), y, random_state=42)
    print(f"X_shape after boruta: {X_trans.shape}")
    print(f"cross val score after boruta {cross_val_score(estimator=rf, X=X_trans, y=y, cv=10).mean()}")

    # rfecv
    X_trans = rfecv_selection(X.copy(), y, random_state=42)
    print(f"X_shape after rfecv: {X_trans.shape}")
    print(f"cross val score after rfecv {cross_val_score(estimator=rf, X=X_trans, y=y, cv=10).mean()}")

    pass
