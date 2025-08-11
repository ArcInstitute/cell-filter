import numba as nb
import numpy as np


@nb.njit
def jit_intersect1d(
    ar1: np.ndarray, ar2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    JIT compiled version of numpy.intersect1d

    Always assumes that the input arrays are unique and always returns the indices
    """
    ar1 = ar1.ravel()
    ar2 = ar2.ravel()

    aux = np.concatenate((ar1, ar2))
    aux_sort_indices = np.argsort(aux, kind="mergesort")
    aux = aux[aux_sort_indices]

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    ar1_indices = aux_sort_indices[:-1][mask]
    ar2_indices = aux_sort_indices[1:][mask] - ar1.size

    return int1d, ar1_indices, ar2_indices
