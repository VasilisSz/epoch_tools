import math
import numpy as np

def row_col_layout(n, rows=None, cols=None):
    """
    Calculate the number of rows and columns required to plot `n` axes.

    Parameters:
        n (int): Total number of axes.
        rows (int, optional): Fixed number of rows. Default is None.
        cols (int, optional): Fixed number of columns. Default is None.

    Returns:
        tuple: A tuple (nrows, ncols) representing 
        the number of rows and columns.
    """
    if rows is not None and cols is not None:
        raise ValueError(
            "Only one of `rows` or `cols` can be specified, not both."
        )

    if rows is not None:
        cols = math.ceil(n / rows)
        return rows, cols

    if cols is not None:
        rows = math.ceil(n / cols)
        return rows, cols

    # If neither rows nor cols is specified, calculate optimal layout
    sqrt_n = math.sqrt(n)
    nrows = math.floor(sqrt_n)
    ncols = math.ceil(n / nrows)

    while nrows * ncols < n:
        nrows += 1
        ncols = math.ceil(n / nrows)

    return nrows, ncols


def compute_err(arr, how):
    """Return SD / SEM / 95 % CI (two-sided) along the first axis."""
    if how is None:
        return None
    if how == "sd":
        return np.nanstd(arr, axis=0, ddof=1)
    if how == "sem":
        return np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    if how == "ci":
        return 1.96 * np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    raise ValueError("err_method must be one of {'sd','sem','ci',None}")