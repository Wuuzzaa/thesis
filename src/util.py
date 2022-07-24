from pathlib import Path


def get_sub_folders(folder: Path) -> list[Path]:
    sub_folders = []

    for path in folder.iterdir():
        if path.is_dir():
            sub_folders.append(path)

    return sub_folders
