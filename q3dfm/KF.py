import numpy as np
from typing import Dict, Union

from q3dfm.long_run_var import long_run_var


def KF(
    Y: np.ndarray, A: np.ndarray, HJ: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> Dict[str, Union[np.ndarray, float]]:
    """Applies fast Kalman filter (see Durbin Koopman (2012))

    Args:
        Y (np.ndarray): k-by-nobs matrix of input data
        A (np.ndarray): r-by-r transition matrix
        HJ (np.ndarray): k-by-r observation matrix
        Q (np.ndarray): r-by-r covariance matrix for transition equation
            residuals
        R (np.ndarray): k-by-1 vector of variances for shocks to observations

    Returns:
        Dict[str, Union[np.ndarray, float]]: dictionary containing:
            Zm : np.ndarray
                m-by-nobs matrix, prior/predicted factor state vector (Z_t|t-1)
            ZmU : np.ndarray
                m-by-(nobs+1) matrix, posterior/updated state vector (Z_t|t)
            Vm : np.ndarray
                m-by-m-by-nobs array, prior/predicted covariance
            VmU : np.ndarray
                m-by-m-by-(nobs+1) array, posterior/updated covariance
            loglik : float
                value of likelihood function
            k_t : np.ndarray
                Kalman gain for the last observation
            UD : np.ndarray
                factor updates for each time step
    """

    # Initialize output values
    sA = A.shape[1]  # number of states
    k, nobs = Y.shape  # number of observations

    # Output dictionary to hold results
    S = {
        "Zm": np.full((sA, nobs), np.nan),  # Prior (Z_t | t-1)
        "Vm": np.full((sA, sA, nobs), np.nan),  # Prior covariance (V_t | t-1)
        "ZmU": np.full((sA, nobs + 1), np.nan),  # Posterior (Z_t | t)
        "VmU": np.full((sA, sA, nobs + 1), np.nan),  # Posterior covariance
        # (V_t | t)
        "loglik": 0,  # Log likelihood initialization
        "UD": np.zeros((sA, k, nobs)),  # Factor updates
    }

    # Initial values following Hamilton (1994)
    Z = np.zeros((sA, 1))  # Z_0|0
    V = long_run_var(A, Q)  # V_0|0

    # Store initial values
    S["ZmU"][:, 0] = Z[:, 0]  # store pre-sample factors
    S["VmU"][:, :, 0] = V  # store pre-sample variance

    # KALMAN FILTER PROCEDURE -------------------------------------------------
    for t in range(nobs):
        # CALCULATING PRIOR DISTRIBUTION --------------------------------------

        # Use transition eqn to create prior estimate for factor
        # i.e. Z = Z_t|t-1
        Z = A @ Z  # prediction for factors in next period

        # Prior covariance matrix of Z (i.e. V = V_t|t-1)
        # Var(Z) = Var(A*Z + u_t) = Var(A*Z) + Var(\epsilon) =
        # A*Vu*A' + Q
        V = A @ V @ A.T + Q  # variance of factor prediction
        V = (V + V.T) / 2  # Trick to make symmetric

        # Store covariance and observation values for t-1 (priors)
        S["Zm"][:, t] = Z  # store predicted factors
        S["Vm"][:, :, t] = V  # store variance of prediction

        # CALCULATING POSTERIOR DISTRIBUTION ----------------------------------

        y_t = Y[:, t]  # extract observatoins in the current period
        ix = np.where(~np.isnan(y_t))[0]  # observed values in y_t
        # Check if y_t contains no data. If so, replace Zu and Vu with prior.

        if len(ix) > 0:
            for j in ix:
                y_j = y_t[j]  # Scalar data for the jth observation
                h_j = HJ[j, :]  # Corresponding row of HJ matrix
                s = h_j @ V @ h_j.T + R[j]  # Variance of forecast for y_j
                pe = y_j - h_j @ Z  # Prediction error
                K = V @ h_j.T / s  # Kalman gain
                Z = Z + K * pe  # Update estimate
                V = V - K[:, None] @ h_j[None, :] @ V  # Update variance
                S["loglik"] -= (
                    np.log(2 * np.pi) + np.log(s) + pe**2 / s
                ) / 2  # Update log likelihood
                S["UD"][:, j, t] = (K * pe).flatten()  # Store update
                # contributions

        # Store posterior values
        S["ZmU"][:, t + 1] = Z[:, 0]  # Store updated factors
        S["VmU"][:, :, t + 1] = V  # Store updated variance

    # Store Kalman gain for the last period (smoothing)
    ix = np.where(~np.isnan(Y[:, nobs - 1]))[0]
    if len(ix) == 0:
        S["k_t"] = np.zeros((sA, sA))
    else:
        H_t = HJ[ix, :]
        C = V @ H_t.T
        P = H_t @ C + np.diag(R[ix])
        K = np.linalg.solve(P, C.T).T
        S["k_t"] = K @ H_t

    return S
