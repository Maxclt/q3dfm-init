# flake8: noqa

import numpy as np
from typing import List, Tuple
from scipy.sparse.linalg import eigs


def InitCond(
    X: np.ndarray,
    m: int,
    p: int,
    frq: np.ndarray,
    isdiff: List[bool],
    blocks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Given standardized data and model information, InitCond() creates
    initial parameter estimates. These are intial inputs in the EM
    algorithm, which re-estimates these parameters using Kalman filtering
    techniques.

    Draw missing value from N(0,1)

        Args:
            X (np.ndarray): Standardized data.
            m (int): Number of common factors.
            p (int): Number of lags in transition equation.
            frq (np.ndarray): frequency mix
            isdiff (List[bool]): logical, is the series differenced
            blocks (np.ndarray): zero restrictions on loadings

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A:   Transition matrix
                H:   Observation matrix
                Q:   Covariance for transition equation residuals
                R:   Covariance for observation equation residuals
    """
    T, k = X.shape
    xBal = np.zeros((T, k))

    # Fill in missing values via cubic spline
    xBal[np.isnan(X)] = np.random.randn(np.isnan(xBal).sum())

    H = np.zeros((k, m))  # Initialize loadings
    xTemp = xBal.copy()
    z = np.zeros((T, m))  # Initialize factors

    for j in range(m):
        lblock = blocks[:, j].astype(
            bool
        )  # Loading restrictions for this factor
        _, G = eigs(
            np.cov(xTemp[:, lblock], rowvar=False), k=1, which="LM"
        )  # Initial guess for loadings
        H[lblock, j] = G  # G is the eigenvector which will be our loading
        z[:, j] = xTemp[:, lblock] @ G  # z is our factor
        xTemp -= (
            z[:, j][:, np.newaxis] @ H[:, j][np.newaxis, :]
        )  # Remove explained variance

    # VAR for transition equation
    Z = np.hstack(
        [z[p - i - 1 : -i - 1, :] for i in range(p)]
    )  # TODO not sure
    sV = Z.shape[1]
    B = (z[p:T, :].T @ Z) @ np.linalg.inv(
        Z.T @ Z + np.eye(sV)
    )  # Transition matrix
    E = z[p:T, :] - Z @ B.T  # Shocks to factors

    # Adjusting for differenced low frequency data
    # -- This tells us how many lags we need in the case of mixed frequencies --
    lags = frq.copy()
    lags[isdiff] = np.array([2 * x - 1 for x in frq[isdiff]])
    pp = max(lags.max(), p)
    # ------------------------------------
    sA = m * pp  # Size of A matrix
    Q = np.zeros(
        (sA, sA)
    )  # Dimensions must line up: shocks to transition equation
    Q[:m, :m] = E.T @ E / (T - p)

    # Shocks to observations
    E = xBal - z @ H.T
    E = E[1:T, :]  # Remove the first row
    R = np.mean(E**2, axis=0) + 1  # Shocks to observation equation

    if pp > p:
        B = np.hstack(
            (B, np.zeros((m, m * (pp - p))))
        )  # Ensure dimensions line up

    A = np.zeros((sA, sA))
    A[: m * pp, : m * pp] = np.vstack(
        [
            B,
            np.hstack([np.eye(m * (pp - 1)), np.zeros((m * (pp - 1), m))]),
        ]
    )  # DFM specified for the companion form of the transition matrix

    return A, H, Q, R
