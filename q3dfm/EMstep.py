import numpy as np
from typing import Tuple, List

from q3dfm.get_HJ import get_HJ
from q3dfm.runKF import runKF
from q3dfm.helper_mat import helper_mat


def EMstep(
    Y: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    p: int,
    frq: np.ndarray,
    is_diff: List[bool],
    blocks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """EMstep reestimates parameters based on the Estimation Maximization (EM)
        algorithm. This is a two-step procedure:

        (1) E-step: the expectation of the log-likelihood is calculated using
        previous parameter estimates.
        (2) M-step: Parameters are re-estimated through the maximisation of the
        log-likelihood (maximize result from (1)).

        See "Maximum likelihood estimation of factor models on data sets with
        arbitrary pattern of missing data" for details about parameter
        derivation (Banbura & Modugno, 2010). This procedure is in much the
        same spirit.

            Args:
                Y (np.ndarray): Series data
                A (np.ndarray): Transition matrix
                H (np.ndarray): Observation matrix
                Q (np.ndarray): Covariance for transition equation residuals
                R (np.ndarray): Covariance for observation matrix residuals
                p (int): Number of lags in transition equation
                frq (np.ndarray): frequency of each series in y
                isdiff (List[bool]): logical (T/F) indicating if series in y is
                    differenced


            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
                    H_new: Updated observation matrix
                    R_new: Updated covariance matrix for residuals of
                        observation matrix
                    A_new: Updated transition matrix
                    Q_new: Updated covariance matrix for residuals for
                        transition matrix
                    loglik: Log likelihood

    References:
        "Maximum likelihood estimation of factor models on data sets with
        arbitrary pattern of missing data" by Banbura & Modugno (2010).
        Abbreviated as BM2010
    """

    # Initialize preliminary values

    # Store series/model values
    k, T = Y.shape
    _, m = H.shape
    sA = A.shape[0]
    pp = sA / m

    # ESTIMATION STEP: Compute the (expected) sufficient statistics for a
    # single Kalman filter sequence

    # Running the Kalman filter and smoother with current parameters
    # Note that log-liklihood is NOT re-estimated after the runKF step: This
    # effectively gives the previous iteration's log-likelihood
    # For more information on output, see runKF

    HJ = get_HJ(H, frq, is_diff, p)

    Zsmooth, Vsmooth, VVsmooth, loglik, _ = runKF(Y, A, HJ, Q, R)
    # Vsmooth gives the variance of contemporaneous factors
    # VVsmooth gives the covariance of factors at one lag for Watson Engle
    # adjustments

    # Normalize
    scl = np.mean(
        Zsmooth[:m, :] ** 2, axis=1
    )  # scaling makes sure factors do not explode
    scl = np.diag(scl**-0.5)
    sscl = np.kron(np.eye(p), scl)
    Scl = np.kron(np.eye(pp), scl)
    Zsmooth = np.dot(Scl, Zsmooth)

    # MAXIMIZATION STEP (TRANSITION EQUATION)
    # See (Banbura & Modugno, 2010) for details.

    # 2A. UPDATE FACTOR PARAMETERS ----------------------------
    # Initialize output
    A_new = A.copy()  # only replace B
    Q_new = (
        Q.copy()
    )  # only replace q (i.e. covariance for contemporaneous shocks)
    mp = m * p  # size of B

    # ESTIMATE FACTOR PORTION OF Q, A
    # Note: EZZ, EZZ_BB, EZZ_FB are parts of equations 6 and 8 in BM 2010

    # E[Z_t*Z_t' | Omega_T]
    # Variance of contemporaneous factors P in the notes
    EZZ = (
        np.dot(Zsmooth[:mp, 1:], Zsmooth[:mp, 1:].T)
        + sscl @ np.sum(Vsmooth[:mp, :mp, 1:], axis=2) @ sscl
    )  # WE adjustment

    # scl is in there because we didn't scale things prior to running the
    # filter/smoother

    EZZ = (EZZ + EZZ.T) / 2  # just to avoid rounding error

    # E[Z_{t-1}*Z_{t-1}' | Omega_T]
    # P_{t-1|T} in the notes
    EZZ_BB = (
        np.dot(Zsmooth[:mp, :-1], Zsmooth[:mp, :-1].T)
        + sscl @ np.sum(Vsmooth[:mp, :mp, :-1], axis=2) @ sscl
    )  # WE adjustment

    EZZ_BB = (EZZ_BB + EZZ_BB.T) / 2

    # E[X_t*Z_{t-1}' | Omega_T]
    # sum of C_{t|T} in the notes
    EZZ_FB = (
        np.dot(Zsmooth[:m, 1:], Zsmooth[:mp, :-1].T)
        + scl @ np.sum(VVsmooth[:m, :mp, :], axis=2) @ sscl
    )  # WE adjustment

    # Equation 6: Estimate VAR(p) for factor
    A_new[:m, :mp] = np.dot(EZZ_FB, np.linalg.inv(EZZ_BB))  # VAR coefficients

    # Equation 8: Covariance matrix of residuals of VAR
    # same as slide 50 in dfm slides
    q = (
        EZZ[:m, :m] - np.dot(A_new[:m, :mp], EZZ_FB.T)
    ) / T  # shortcut --- very clever (again)
    Q_new[:m, :m] = (q + q.T) / 2

    # 3 MAXIMIZATION STEP (observation equation)

    # INITIALIZATION AND SETUP ----------------------------------------------
    H_new = H.copy()  # Loadings

    for j in range(k):  # Loop through observables
        fq = frq[j]  # get the frequency of this series
        y = Y[j, :]  # the actual observed data
        y_idx = ~np.isnan(y)  # find values that are observed
        y_obs = y[y_idx]
        lblock = blocks[j, :].astype(
            bool
        )  # which factors can this series load on?

        if fq == 1:
            Z_obs = Zsmooth[
                :m, 1:
            ]  # drop pre-sample value Z_0, columns are time in y
            Z_obs = Z_obs[lblock, :][
                :, y_idx
            ]  # Z_obs where y observed for those factors y can load on
            # sum of P^x_{t|T} in the slides
            V_obs = np.sum(
                Vsmooth[:m, :m, np.concatenate([[False], y_idx])], axis=2
            )  # Vsmooth where y observed
        else:
            J = helper_mat(fq, is_diff[j], m, sA)
            Z_obs = np.dot(J, Zsmooth[:sA, 1:])
            Z_obs = Z_obs[lblock, :][:, y_idx]
            V_obs = (
                J
                @ np.sum(
                    Vsmooth[:sA, :sA, np.concatenate([[False], y_idx])], axis=2
                )
                @ J.T
            )
        V_obs = (
            scl[lblock, :][:, lblock] @ V_obs @ scl[lblock, :][:, lblock]
        )  # for zero restrictions
        EZZ = np.dot(Z_obs, Z_obs.T) + V_obs
        EZZ = (EZZ + EZZ.T) / 2  # get rid of rounding error
        h = (
            np.dot(y_obs, Z_obs.T) / EZZ
        )  # almost OLS, just adjusting for the fact that factors are
        # estimated not observed
        H_new[j, lblock] = h  # plug estimated values of h in
        R[j] = (
            np.sum((y_obs - np.dot(h, Z_obs)) ** 2)
            + np.dot(np.dot(h, V_obs), h.T)
        ) / y_obs.shape[
            0
        ]  # shocks to this series in observation equation

    R_new = R.copy()

    return H_new, R_new, A_new, Q_new, loglik
