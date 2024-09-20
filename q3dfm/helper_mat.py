import numpy as np


def helper_mat(fq: int, is_diff: bool, r: int, m: int) -> np.ndarray:
    # TODO check if r, m is right
    """Return helper matrix J for observation equation

    Args:
        fq (int): frequency of serie j
        is_diff (bool): True/False if serie j is differenciate
        r (int): number of rows of J
        m (int): number of columns of J

    Returns:
        np.ndarray: helper matrix J of size r-by-m
    """
    J = np.zeros((r, m))
    if is_diff:
        J[:, : r * (2 * fq - 1)] = np.kron(
            np.concatenate((np.arange(1, fq + 1), np.arange(fq - 1, 0, -1)))
            / fq,
            np.eye(r),
        )

    else:
        J[:, : r * fq] = np.kron(np.ones((1, fq)) / fq, np.eye(r))

    return J
