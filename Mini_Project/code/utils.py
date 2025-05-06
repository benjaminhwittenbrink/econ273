import os
import pickle
import numpy as np
import pandas as pd

from typing import List, Dict, Optional, Tuple, Any


## I/O functions
def read_pickle(file_path: str) -> object:
    """
    Read a pickle file and return the object.
    """
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def read_file(file_path: str) -> object:
    """
    Read a text file and return its content.
    """
    if ".pkl" in file_path:
        return read_pickle(file_path)
    elif ".csv" in file_path:
        return pd.read_csv(file_path)
    elif ".txt" in file_path:
        with open(file_path, "r") as f:
            content = f.read()
        return content
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def load_data(
    folders: List[str], main_dir: Optional[str] = None
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """
    Load data from the specified folders.
    """
    DDobjs = {}
    DDdata = {}
    for folder in folders:
        for file in os.listdir(os.path.join(main_dir, folder)):
            if file.endswith(".pkl"):
                DDobjs[folder] = read_file(os.path.join(main_dir, folder, file))
            elif file.endswith(".csv"):
                DDdata[folder] = read_file(os.path.join(main_dir, folder, file))
    return DDobjs, join_data(DDdata)


def join_data(res: Dict[str, object]) -> pd.DataFrame:
    """
    Join data from the specified folders into a single DataFrame.
    """
    data = []
    for key in res:
        data.append(res[key])
    data = pd.concat(data, axis=0)
    data = data.reset_index(drop=True)
    return data


# def dict_convert_lists_to_arrays(d):
#     for key in d:
#         if isinstance(d[key], list):
#             d[key] = np.array(d[key])
#     return d


# def convergence_check(x, x_new, tol):
#     """
#     Check for convergence.
#     """
#     return np.max(np.abs(x_new - x)) < tol
