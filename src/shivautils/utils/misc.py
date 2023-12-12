"""
Miscellaneous functions usefull in multiple scripts
"""

import hashlib
import os
import pathlib


def md5(fname):
    """
    Create a md5 hash for a file or a folder

    Args:
        fname (str): file/folder path

    Returns:
        str: hexadecimal hash for the file/folder
    """
    hash_md5 = hashlib.md5()
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    elif os.path.isdir(fname):
        fpath = pathlib.Path(fname)
        file_list = [f for f in fpath.rglob('*') if os.path.isfile(f)]
        file_list.sort()
        for sub_file in file_list:
            with open(sub_file, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()
