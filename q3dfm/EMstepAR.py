import numpy as np
from typing import Tuple, List

from q3dfm.runKF import runKF
from q3dfm.helper_mat import helper_mat
from q3dfm import get_HJ


def EMstepAR(
    Y: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    p: int,
    frq: List[int],
    isdiff: List[bool],
    blocks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Applies EM algorithm for parameter reestimation.

    Args:
        Y (np.ndarray): Series data (observations) of shape (k, T).
        A (np.ndarray): Transition matrix.
        H (np.ndarray): Observation matrix.
        Q (np.ndarray): Covariance matrix for transition equation residuals.
        R (np.ndarray): Covariance matrix for observation matrix residuals.
        p (int): Number of lags in the transition equation.
        frq (List[int]): List of frequencies for each series.
        isdiff (List[bool]): List indicating if the series is differenced.
        blocks (np.ndarray): Block structure for each series (i.e., loadings
            on factors).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
            H_new (ndarray): Updated observation matrix.
            R_new (ndarray): Updated covariance matrix for observation matrix
                residuals.
            A_new (ndarray): Updated transition matrix.
            Q_new (ndarray): Updated covariance matrix for transition matrix
                residuals.
            loglik (float): Log-likelihood.
    """

    # Initialize preliminary values
    k, T = Y.shape
    m = H.shape[1]
    sA = A.shape[0] - k  # Exclude AR error terms

    # ESTIMATION STEP: Compute the expected sufficient statistics
    # Running Kalman filter and smoother with current parameters
    HJ = get_HJ(H, frq, isdiff, p)
    HH = np.hstack((HJ, np.eye(k)))  # Combine HJ with identity matrix

    Zsmooth, Vsmooth, VVsmooth, loglik = runKF(Y, A, HH, Q, R)

    # Normalize
    scl = np.mean(Zsmooth[:m, :] ** 2, axis=1)
    scl = np.diag(scl**-0.5)
    sscl = np.kron(np.eye(p), scl)
    Scl = np.kron(np.eye(sA // m), scl)
    Zsmooth[:sA, :] = Scl @ Zsmooth[:sA, :]

    # MAXIMIZATION STEP (TRANSITION EQUATION)
    A_new = A.copy()
    Q_new = Q.copy()
    mp = m * p

    # Estimate VAR(p) for factor
    EZZ = (
        Zsmooth[:mp, 1:] @ Zsmooth[:mp, 1:].T
        + sscl @ np.sum(Vsmooth[:mp, :mp, 1:], axis=2) @ sscl
    )
    EZZ_BB = (
        Zsmooth[:mp, :-1] @ Zsmooth[:mp, :-1].T
        + sscl @ np.sum(Vsmooth[:mp, :mp, :-1], axis=2) @ sscl
    )
    EZZ_FB = (
        Zsmooth[:m, 1:] @ Zsmooth[:mp, :-1].T
        + scl @ np.sum(VVsmooth[:m, :mp, :], axis=2) @ sscl
    )

    # Equation 6: Estimate VAR(p) coefficients
    A_new[:m, :mp] = EZZ_FB @ np.linalg.inv(EZZ_BB)

    # Equation 8: Covariance matrix of residuals of VAR
    Q_new[:m, :m] = (EZZ[:m, :m] - A_new[:m, :mp] @ EZZ_FB.T) / T

    # Update AR(1) components for each series
    for j in range(k):
        EZZ = np.sum(Zsmooth[sA + j, 1:] ** 2) + np.sum(
            Vsmooth[sA + j, sA + j, 1:], axis=2
        )
        EZZ_BB = np.sum(Zsmooth[sA + j, :-1] ** 2) + np.sum(
            Vsmooth[sA + j, sA + j, :-1], axis=2
        )
        EZZ_FB = Zsmooth[sA + j, 1:] @ Zsmooth[sA + j, :-1].T + np.sum(
            VVsmooth[sA + j, sA + j, :], axis=2
        )

        # Equation 6: Update VAR coefficients for AR(1)
        A_new[sA + j, sA + j] = EZZ_FB / EZZ_BB
        # Update covariance matrix of residuals for AR(1)
        Q_new[sA + j, sA + j] = (EZZ - A_new[sA + j, sA + j] * EZZ_FB) / T

    # MAXIMIZATION STEP (OBSERVATION EQUATION)
    H_new = H.copy()

    for j in range(k):
        fq_j = frq[j]
        y = Y[j, :] - Zsmooth[sA + j, 1:]
        y_idx = ~np.isnan(y)
        y_obs = y[y_idx]
        lblock = blocks[j, :].astype(bool)

        if fq_j == 1:
            Z_obs = Zsmooth[:m, 1:]
            Z_obs = Z_obs[lblock, :][:, y_idx]
            V_obs = np.sum(
                Vsmooth[:m, :m, np.hstack(([False], y_idx))], axis=2
            )
        else:
            J = helper_mat(fq_j, isdiff[j], m, sA)
            Z_obs = J @ Zsmooth[:sA, 1:]
            Z_obs = Z_obs[lblock, :][:, y_idx]
            V_obs = (
                J
                @ np.sum(
                    Vsmooth[:sA, :sA, np.hstack(([False], y_idx))], axis=2
                )
                @ J.T
            )
        V_obs = scl[lblock, :][:, lblock] @ V_obs @ scl[lblock, :][:, lblock]
        V_ar = np.sum(
            Vsmooth[sA + j, sA + j, np.hstack(([False], y_idx))], axis=2
        )
        EZZ = Z_obs @ Z_obs.T + V_obs
        h = (y_obs @ Z_obs.T) @ np.linalg.inv(EZZ)
        H_new[j, lblock] = h
        R[j] = (
            (y_obs - h @ Z_obs) @ (y_obs - h @ Z_obs).T
            + h @ V_obs @ h.T
            + V_ar
        ) / len(y_obs)

    R_new = R.copy()

    return H_new, R_new, A_new, Q_new, loglik
