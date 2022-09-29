import pandas as pd
from pycaret.classification import *
from sklearn.metrics import accuracy_score


def run_pycaret(X_train:pd.DataFrame, X_test:pd.DataFrame, y_train, y_test, random_state, model_types_to_use, cv, n_iter_tune):
    """
    Function to run pycaret for classification tasks. It setups an experiment, train, tune and stack the wanted models.

    :param n_iter_tune: int amount of iterations for hyperparameter search.
    :param X_train: Dataframe
    :param X_test: Dataframe
    :param y_train: Series?
    :param y_test: Series?
    :param random_state: int
    :param model_types_to_use: list strings of classifiers to train. see pycaret docu: https://pycaret.gitbook.io/docs/get-started/functions/others#models
    :param cv: int how much foldes to use
    :return: dataframe with the accuracy scores of each classifier descending sorted
    """
    # store the featurenames WITHOUT target("y") column
    feature_names = list(X_train.columns)

    # pycaret wants the target in a dataframe column
    X_train["y"] = y_train
    X_test["y"] = y_test

    print("---")
    print("SETUP")
    print("---")

    # setup pycaret
    experiment = setup(
        data=X_train,
        target="y",
        test_data=X_test,
        preprocess=False,
        # data_split_shuffle=False,
        n_jobs=-1,
        fold=cv,
        fold_shuffle=True,
        # must be set because of this bug which leads from pycaret is not adjustd to sklearn version 1? ValueError: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True. Info from here: https://stackoverflow.com/questions/67728802/valueerror-setting-a-random-state-has-no-effect-since-shuffle-is-false-you-sho
        numeric_features=feature_names,
        # force all features to be regarded as numeric not categorical which avoids problems with xgboost and lightgbm
        session_id=random_state,
        html=False,  # must be set to False when run outside of a notebook (ipython)
        silent=True,  # ignore the prompt input after the setup
    )

    # train models
    print("---")
    print("TRAIN MODELS")
    print("---")

    selected_models = compare_models(
        include=model_types_to_use,
        errors="raise",
        n_select=len(model_types_to_use)  # all models included
    )

    # predict models (untuned)
    print("---")
    print("PREDICT MODELS (UNTUNED)")
    print("---")

    model_names = []
    accuracy_scores = []

    for model in selected_models:
        model_names.append(model.__class__.__name__)
        prediction_df = predict_model(model)
        accuracy_scores.append(accuracy_score(prediction_df["Label"], prediction_df["y"].astype("int64")))

    predict_df = pd.DataFrame(data={"model": model_names, "accuracy": accuracy_scores}).sort_values(by="accuracy",
                                                                                                    ascending=False).reset_index(
        drop=True)

    # tune models
    print("---")
    print("TUNE MODELS")
    print("---")

    tuned_models = []
    for model in selected_models:
        tuned_model = tune_model(model, choose_better=True, n_iter=n_iter_tune)
        tuned_models.append(tuned_model)

    model_names = []
    accuracy_scores = []

    # predict with tuned models
    print("---")
    print("PREDICT MODELS TUNED")
    print("---")

    for model in tuned_models:
        model_names.append(f"{model.__class__.__name__} tuned")
        predict_df_tuned = predict_model(model)
        accuracy_scores.append(accuracy_score(predict_df_tuned["Label"], predict_df_tuned["y"].astype("int64")))

    predict_df_tuned = pd.DataFrame(data={"model": model_names, "accuracy": accuracy_scores}).sort_values(by="accuracy",
                                                                                                          ascending=False).reset_index(
        drop=True)

    # stack tuned models
    print("---")
    print("STACK MODELS")
    print("---")

    stacked_model = stack_models(
        estimator_list=tuned_models,
        round=6,
        restack=False,
        # False means only the Predictions of Layer 0 are used to train the final estimator. True uses the basefeatures too.
        choose_better=False
    )

    # predict stacked model
    print("---")
    print("PREDICT STACKED MODELS")
    print("---")

    predict_df_stacked = predict_model(stacked_model, round=6)
    predict_df_stacked = pd.DataFrame(data={"model": [f"{stacked_model.__class__.__name__} tuned"], "accuracy": [
        accuracy_score(predict_df_stacked["Label"], predict_df_stacked["y"].astype("int64"))]})

    # make pycaret results dataframe
    caret_results_df = pd.concat([predict_df, predict_df_tuned, predict_df_stacked]).\
        sort_values(by="accuracy", ascending=False).\
        reset_index(drop=True)

    return caret_results_df


