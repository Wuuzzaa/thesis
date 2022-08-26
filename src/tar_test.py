from constants import *
from shutil import make_archive, unpack_archive

# runs but only one core is used
format_string = "bztar" #  "xztar"
work_folder = Path(r"C:\Users\jonas\PycharmProjects\thesis\data")
path_folder_to_archive = work_folder.joinpath("datasets")
path_archive_folder = work_folder.joinpath("datasets_copy")
path_archive_file = work_folder.joinpath("datasets_copy.tar.bz2") #  work_folder.joinpath("datasets_copy.tar.xz")


# make archive
make_archive(
    base_name=str(path_archive_folder),
    format=format_string,
    root_dir=path_folder_to_archive
)

# unpack archive
unpack_archive(
    filename=str(path_archive_file),
    extract_dir=str(path_archive_folder),
    format=format_string
)