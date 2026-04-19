"""IO Module"""

import numpy as np
import os
from typing import Union

def load_data(
        file_path: str, 
        delimiter:str =",",
        skip_header:int = 0,
    )-> np.ndarray:
    """Loads CSV data
    
    Args: 
        file_path (str): Path to the CSV file.
        delimiter (str): Placeholder on which data has to be separated.
        skip_header (int): Row containing columns headers.
    
    Returns:
        Loaded data in numpy arrays format.

    Raises: 
        FileNotFoundError: Error raised when `file_path` does not point to
            existing file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"file not found — '{file_path}'."
        )

    data = np.genfromtxt(
        file_path, 
        delimiter=delimiter, 
        skip_header=skip_header,
        # We have to confirm whether we only have to handle the numeric data type
        # in the columns of the loaded data type. 
    )

    return data