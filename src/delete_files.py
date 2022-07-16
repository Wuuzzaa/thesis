from pathlib import Path


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
    root_folder = Path("..//data")

    #filename = "pca_train_mle_clean.feather"
    #filename = "pca_test_mle_clean.feather"
    #filename = "kpca_train_clean.feather"
    #filename = "kpca_test_clean.feather"
    #filename = "pca_train_clean.feather"
    #filename = "pca_test_clean.feather"

    delete_files(root_folder, filename)