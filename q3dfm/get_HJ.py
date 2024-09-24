import numpy as np


def get_HJ(
    H: np.ndarray, frq: np.ndarray, is_diff: bool, p: int
) -> np.ndarray:
    """Get loadings integrating helper matrix J for mixed frequency data

    Args:
        H (np.ndarray): contemporaneous loadings
        frq (np.ndarray): frequency of series (keyed to index)
        isdiff (bool): logical, is the series differenced (keyed to index)
        p (int): number of lags in transition equation

    Returns:
        np.ndarray: returns Delta matrix
    """
    k, m = H.shape
    lags = np.copy(frq)
    lags[is_diff] = 2 * frq[is_diff] - 1
    pp = int(max(max(lags), p))
    HJ = np.zeros((k, m * pp))
    HJ[:, :m] = H
    idx = [i for i, x in enumerate(frq) if x > 1]
    for j in idx:
        if is_diff(j):
            HJ[j, : m * (2 * frq(j) - 1)] = np.dot(
                H[j, :],
                np.kron(
                    np.concatenate(
                        [
                            np.arange(1, frq[j] + 1),
                            np.arange(frq[j] - 1, 0, -1),
                        ]
                    )
                    / frq[j],
                    np.eye(m),
                ),
            )
        else:
            HJ[j, : m * frq[j]] = np.dot(
                H[j, :], np.kron(np.ones(frq[j]) / frq[j], np.eye(m))
            )
    return HJ
