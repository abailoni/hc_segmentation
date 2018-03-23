from numba import njit
import numpy as np

@njit
def find_first_index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    return None