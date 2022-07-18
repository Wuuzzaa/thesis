from pathlib import Path
from constants import *


def delete_files(root_folder: Path, filename: str):
    """
    All child folders of the given root folder are traversed in search for files with the given filename to delete


    :param root_folder: Path of the rootfolder.
    :param filename: filename to delete
    :return:
    """
    to_delete = []

    # travers root folder
    for path in root_folder.rglob("*"):
        if path.is_file():
            if path.name == filename:
                to_delete.append(path)

    # delete
    for path in to_delete:
        print(f"delete file: {path}")
        path.unlink()


if __name__ == "__main__":
    from constants import *
    root_folder = DATA_FOLDER_PATH

    # set a filename
    filenames = [
        # X and y
        # X_CLEAN_FILE_NAME,
        # y_FILE_NAME,

        # pca
        X_TRAIN_CLEAN_PCA_FILE_NAME,
        X_TEST_CLEAN_PCA_FILE_NAME,

        # kpca
        X_TRAIN_CLEAN_KPCA_FILE_NAME,
        X_TEST_CLEAN_KPCA_FILE_NAME,

        # results file
        RESULTS_DATAFRAME
    ]

    # traverse the root folder and subfolders and delete all matches
    for filename in filenames:
        print(f"delete all files with name: {filename}")
        delete_files(root_folder, filename)