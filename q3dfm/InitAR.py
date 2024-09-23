# flake8: noqa

import numpy as np
from typing import Tuple, List

from q3dfm.get_HJ import get_HJ
from q3dfm.runKF import runKF


def init_ar(
    Y: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    frq: List[int],
    isdiff: List[bool],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """InitCondAR() calculates initial estimates for autoregressive error
    terms and shocks to the observation and transition equation.

    Args:
        Y (np.ndarray): Standardized data
        A (np.ndarray): Transition matrix from model without AR errors
        H (np.ndarray): Loadings for model without AR errors
        Q (np.ndarray): Shocks to the transition equation for model without AR errors
        R (np.ndarray): Shocks to the observation equation for model without AR errors
        frq (List[int]): Frequency mix (specific implementation not provided)
        isdiff (List[bool]): Is the data differenced

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A_out: np.ndarray - Transition matrix with AR parameters
            H: np.ndarray - Observation matrix (unchanged from input)
            Q_out: np.ndarray - Covariance for transition equation residuals
            R: np.ndarray - Covariance for observation equation residuals
    """

    k, T = Y.shape
    sA = A.shape[0]
    p = sA // H.shape[1]

    HJ = get_HJ(H, frq, isdiff, p)
    Zsmooth = runKF(Y, A, HJ, Q, R)

    yBal = np.zeros((T, k))

    # Fill in missing values following N(0,1)
    yBal[np.isnan(Y)] = np.random.randn(np.isnan(yBal).sum())

    # Shocks to observations
    E = yBal - (HJ @ Zsmooth[:, 1:].T)

    # AR components of transition matrix
    a = np.zeros(k)
    for j in range(k):
        a[j] = (
            E[1:T, j].T
            @ E[0 : T - 1, j]
            / (E[0 : T - 1, j].T @ E[0 : T - 1, j])
        )

    A_out = np.block([[A, np.zeros((sA, k))], [np.zeros((k, sA)), np.diag(a)]])
    # AR error terms

    E = (
        E[1:T, :]
        - np.repeat(a[np.newaxis, :], T - 1, axis=0) * E[0 : T - 1, :]
    )  # Idiosyncratic errors
    R = np.mean(E**2, axis=0)  # Variance of idiosyncratic errors

    Q_out = np.zeros((sA + k, sA + k))
    Q_out[0:sA, 0:sA] = Q
    Q_out[sA : sA + k, sA : sA + k] = np.diag(R)

    return A_out, H, Q_out, R
