import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomTreesEmbedding, StackingClassifier
from sklearn.model_selection import cross_val_predict

from constants import *
import numpy as np
from shutil import make_archive, unpack_archive
import time
from time import perf_counter
import warnings

from src.calc_scores import get_X_train_X_test_y_train_y_test

dataset_id = "3"
#dataset_id = "40923" # 1489 # 3

# load data
X_train, X_test, y_train, y_test = get_X_train_X_test_y_train_y_test(dataset_folder=DATASETS_FOLDER_PATH.joinpath(dataset_id), random_state=RANDOM_STATE, X_file_name=X_FILTERED_FILE_NAME, y_file_name=Y_FILE_NAME)

# short feedback of the data and classes
print(f"X_train shape: {X_train.shape}")
print(f"target classes: \n{y_train.value_counts()}")
print(f"total {len(y_train.value_counts())} classes\n")

#sample if needed
sample_size = 10_000

if len(X_train) > sample_size:
    warnings.warn(
        f"Sample is used of {sample_size}")
    X_train_sample = X_train.sample(n=sample_size, random_state=RANDOM_STATE)

else:
    X_train_sample = X_train

cv = 5

estimators = [
    ('rf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ('lr', LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1)),
]

clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1),
    cv=cv,
)

clf.fit(X_train, y_train)
print(f"stacking score: {clf.score(X_test, y_test)}")


est = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rf_cross_val_predict = cross_val_predict(est, X_train, y_train, cv=cv, method="predict_proba")
est.fit(X_train, y_train)
rf_test_predict = est.predict_proba(X_test)

est = LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1)
est.fit(X_train, y_train)
lr_test_predict = est.predict_proba(X_test)
lr_cross_val_predict = cross_val_predict(est, X_train, y_train, cv=cv, method="predict_proba")

X_train_final = np.concatenate((rf_cross_val_predict, lr_cross_val_predict), axis=1)
X_test_final = np.concatenate((rf_test_predict, lr_test_predict), axis=1)

est = LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1)
est.fit(X_train_final, y_train)
print(f"score stacking by hand: {est.score(X_test_final, y_test)}")

pass