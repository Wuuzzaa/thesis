import math
from time import time
import pandas as pd
import keras
from keras import layers, regularizers
import tensorflow as tf


def _create_autoencoder_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    params: dict,
    prefix: str,
    random_state: int,
):
    # sample if needed
    #sample_size = 10_000
    #
    # if len(X_train) > sample_size:
    #     warnings.warn(
    #         f"Dataset is huge. Kernel PCA needs huge memory with too much rows. PCA gets slow. Sample is used of {sample_size}")
    #     X_train_sample = X_train.sample(n=sample_size, random_state=random_state)
    #
    # else:
    #     X_train_sample = X_train

    # amount features in X
    X_n_features = len(X_train.columns)

    # This is the size of our encoded representations = n new features
    encoding_dim = min(10, int(math.sqrt(X_n_features)))

    # Encoder
    input_layer = keras.Input(shape=(X_n_features,), name="input_layer")
    x = layers.Dense(int(X_n_features / 2), activation=params["activation"], name="hidden_encode", activity_regularizer=regularizers.l1(10e-5))(input_layer)
    x = layers.Dense(encoding_dim, activation=params["activation"], name="encode_layer", activity_regularizer=regularizers.l1(10e-5))(x)
    encoder_model = keras.Model(input_layer, x)

    # Decoder layer
    x = layers.Dense(int(X_n_features / 2), activation=params["activation"], name="hidden_decode", activity_regularizer=regularizers.l1(10e-5))(x)
    x = layers.Dense(X_n_features, activation=params["activation"], name="decode", activity_regularizer=regularizers.l1(10e-5))(x)

    # autoencoder model
    autoencoder = keras.Model(input_layer, x)
    autoencoder.compile(optimizer=params["optimizer"], loss=params["loss"])

    # specify how early stopping works
    callback_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # "loss" -> train loss, "val_loss" -> validation loss
        patience=params["early_stopping_patience"],
        verbose=1,
        restore_best_weights=True
    )

    # fit
    autoencoder.fit(
        X_train,
        X_train,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        shuffle=True,
        validation_split=params["validation_split"],
        # use_multiprocessing=True, # seems to have no effect. afaik tensorflow uses all cores by default on a single pc
        callbacks=[callback_early_stopping],
    )

    # make features for train and test
    X_train_trans = pd.DataFrame(encoder_model.predict(X_train)).add_prefix(prefix)
    X_test_trans = pd.DataFrame(encoder_model.predict(X_test)).add_prefix(prefix)

    return X_train_trans, X_test_trans


if __name__ == "__main__":
    from calc_scores import get_X_train_X_test_y_train_y_test
    from pathlib import Path
    from constants import *

    # load some data
    X_train, X_test, y_train, y_test = get_X_train_X_test_y_train_y_test(
        dataset_folder=Path(r"C:\Users\jonas\PycharmProjects\thesis\data\datasets\40923"),
        random_state=42,
        X_file_name=X_CLEAN_FILE_NAME,
        y_file_name=Y_FILE_NAME,
    )

    X_train_trans, X_test_trans = _create_autoencoder_features(
        X_train=X_train,
        X_test=X_test,
        params=AUTOENCODER_PARAMS,
        prefix="autoencoder_",
        random_state=RANDOM_STATE
    )

    estimator = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE)

    print("#"*80)
    print("BASELINE FEATURES")
    print("#" * 80)

    # start timer
    start_time = time()

    estimator.fit(X_train, y_train)
    print(f"Fit done in {round(time() - start_time, 2)} seconds")

    score = estimator.score(X_test, y_test)
    print(f"score: {round(score, 4)}\n")

    print("#" * 80)
    print("BASELINE FEATURES AND AUTOENCODED FEATURES")
    print("#" * 80)

    # start timer
    start_time = time()

    estimator.fit(pd.concat([X_train, X_train_trans], axis="columns"), y_train)
    print(f"Fit done in {round(time() - start_time, 2)} seconds")

    score = estimator.score(pd.concat([X_test, X_test_trans], axis="columns"), y_test)
    print(f"score: {round(score, 4)}")

    print()
    print("#" * 80)
    print("ONLYAUTOENCODED FEATURES")
    print("#" * 80)

    # start timer
    start_time = time()

    estimator.fit(X_train_trans, y_train)
    print(f"Fit done in {round(time() - start_time, 2)} seconds")

    score = estimator.score(X_test_trans, y_test)
    print(f"score: {round(score, 4)}")

    pass