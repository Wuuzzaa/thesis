import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomTreesEmbedding

from constants import *
import numpy as np
from shutil import make_archive, unpack_archive
import time
from time import perf_counter
import warnings

from src.calc_scores import get_X_train_X_test_y_train_y_test

dataset_id = "40923"
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

# make new features
embedding = RandomTreesEmbedding(random_state=RANDOM_STATE, n_jobs=-1)
embedding.fit(X_train_sample, y_train)
X_train_trans = pd.DataFrame(embedding.transform(X_train).toarray()).add_prefix("random_trees_embedding_")
X_test_trans = pd.DataFrame(embedding.transform(X_test).toarray()).add_prefix("random_trees_embedding_")

print(f"dataset id: {dataset_id}")
print(f"X_train_trans shape: {X_train_trans.shape}")

# run baseline
estimator = HistGradientBoostingClassifier(categorical_features=None, random_state=RANDOM_STATE)
#estimator = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
estimator.fit(X_train, y_train)
baseline_score = estimator.score(X_test, y_test)
print(f"baseline score: {baseline_score}")

# run with new features
estimator.fit(pd.concat([X_train, X_train_trans], axis="columns"), y_train)
combined_score = estimator.score(pd.concat([X_test, X_test_trans], axis="columns"), y_test)
print(f"combined score: {combined_score}")



pass