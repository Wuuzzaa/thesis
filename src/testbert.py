import pandas as pd
import matplotlib.pyplot as plt
from constants import *
import numpy as np
from shutil import make_archive, unpack_archive

# dataset_id = "40923"
# #dataset_id = "40923" # 1489 # 3
#
# # load data
# X_train, X_test, y_train, y_test = get_X_train_X_test_y_train_y_test(dataset_folder=DATASETS_FOLDER_PATH.joinpath(dataset_id), random_state=RANDOM_STATE, X_file_name=X_CLEAN_FILE_NAME, y_file_name=Y_FILE_NAME)
#
# # short feedback of the data and classes
# print(f"X_train shape: {X_train.shape}")
# print(f"target classes: \n{y_train.value_counts()}")
# print(f"total {len(y_train.value_counts())} classes\n")

# sample if needed
# sample_size = 10_000
#
# if len(X_train) > sample_size:
#     warnings.warn(
#         f"Dataset is huge. Kernel PCA needs huge memory with too much rows. PCA gets slow. Sample is used of {sample_size}")
#     X_train_sample = X_train.sample(n=sample_size, random_state=RANDOM_STATE)
#
# else:
#     X_train_sample = X_train

# load results dataframe
df = pd.read_feather(RESULTS_FILE_PATH)
df = df[df.columns[df.columns.str.contains("dataset_id|test_score")]]
df = df[df.columns[df.columns.str.contains("dataset_id|stacking|baseline_filtered_test_score")]]
df = df[df.columns[~df.columns.str.contains(">|only")]]

# make plots and store the accuracy improvement
for index, row in df.iterrows():
    df_plot = row.to_frame().T

    # set dataset id as index
    df_plot = df_plot.set_index("dataset_id")

    # rename the column for better readability
    columns = list(df_plot.columns)
    columns = [column.replace("_", " ").replace("test score", "").rstrip() for column in columns]

    df_plot.columns = columns

    # get dataset id
    dataset_id = df_plot.index.array[0]

    df_plot = df_plot.T.sort_values(by=dataset_id, ascending=False).T

    # adjust color of baseline score
    colors = ["blue"] * len(columns)
    index_baseline = list(df_plot.columns).index("baseline filtered")
    colors[index_baseline] = "red"

    # set figure size
    plt.figure(figsize=(10, 10))

    # horizontal barplot
    plt.barh(
        y=df_plot.columns,
        width=df_plot.values.squeeze(),
        color=colors,
    )

    # add accuracy to the end of bars
    for index, value in enumerate(df_plot.values.squeeze()):
        plt.text(value, index, str(value.round(4)))

    # add title with accuracy performance gain
    highest_accuracy = df_plot.values.squeeze()[0]
    baseline_accuracy = df_plot.values.squeeze()[index_baseline]
    performance_gain_percent = round((highest_accuracy / baseline_accuracy) * 100 - 100, 2)
    plt.title(f"Dataset ID: {dataset_id}. Accuracy improved by {performance_gain_percent}%")

    # make a vertical line for the baseline score
    plt.axvline(x=baseline_accuracy, color='g')

    # create folder for dataset
    dataset_folder = PLOTS_FOLDER_PATH.joinpath(str(int(dataset_id)))
    dataset_folder.mkdir(parents=True, exist_ok=True)

    # store
    file_path = dataset_folder.joinpath("accuracy_improvement_plot")
    plt.savefig(fname="temp_plot", bbox_inches="tight")

pass