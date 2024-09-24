import numpy as np
from typing import Dict, Union


def Ksmooth(
    A: np.ndarray, S: np.ndarray
) -> Dict[str, Union[np.ndarray, float]]:
    """Applies fixed-interval Kalman smoother

    Model:
        Y_t = HJ_t Z_t + e_t for e_t ~ N(0, R)
        Z_t = A Z_{t-1} + mu_t for mu_t ~ N(0, Q)

    Args:
        A (np.ndarray): r-by-r transition matrix where r is the total number
            of factors including lags
        S (np.ndarray): structure returned by Ksmooth()

    Returns:
        Dict[str, Union[np.ndarray, float]]: dictionnary containing S and:
            - ZmT: m-by-(nobs+1) matrix, smoothed states
                (ZmT(:,t+1) = Z_t|T)
            - VmT: m-by-m-by-(nobs+1) array, smoothed factor covariance
                matrices (VmT(:,:,t+1) = V_t|T = Cov(Z_t|T))
            - VmT_1: m-by-m-by-nobs array, smoothed lag 1 factor covariance
                matrices (VmT_1(:,:,t) = Cov(Z_t Z_t-1|T))
    """

    # ORGANIZE INPUT ----------------------------------------------------------

    # initialize output matrices
    m, nobs = S["Zm"].shape
    S["ZmT"] = np.zeros((m, nobs + 1))
    S["VmT"] = np.zeros((m, m, nobs + 1))
    S["VmT_1"] = np.zeros((m, m, nobs + 1))

    # fill the final period of ZmT, VmT with SKF() posterior values
    S["ZmT"][:, nobs] = np.squeeze(S["ZmU"][:, nobs])
    S["VmT"][:, :, nobs] = np.squeeze(S["VmU"][:, :, nobs])

    # Initialize VmT_1 lag 1 covariance matrix for final period
    S["VmT_1"][:, :, nobs - 1] = (
        (np.eye(m) - S["k_t"]) @ A @ np.squeeze(S["VmU"][:, :, nobs - 1])
    )

    # Used for recursion process. See companion file for details
    J_2 = (
        np.squeeze(S["VmU"][:, :, nobs - 1])
        @ A.T
        @ np.linalg.pinv(np.squeeze(S["Vm"][:, :, nobs - 1]))
    )

    # RUN SMOOTHING ALGORITHM -------------------------------------------------

    # Loop through time reverse-chronologically (starting at final period nobs)
    for t in range(nobs - 1, -1, -1):

        # Store posterior and prior factor covariance values
        VmU = np.squeeze(S["VmU"][:, :, t])  # P_t|t
        Vm1 = np.squeeze(S["Vm"][:, :, t])  # P_t|t-1

        # Store previous period smoothed factor covariance and lag-1 covariance
        V_T = np.squeeze(S["VmT"][:, :, t + 1])  # P_t|T
        V_T1 = np.squeeze(S["VmT_1"][:, :, t])  # P_t-1|T for EM algorithm

        # g in the notes
        J_1 = J_2  # elements of smoothing algorithm

        # Update smoothed factor estimate
        S["ZmT"][:, t] = S["ZmU"][:, t] + J_1 @ (
            S["ZmT"][:, t + 1] - A @ S["ZmU"][:, t]
        )  # Equation 22 in the notes

        # Update smoothed factor covariance matrix
        S["VmT"][:, :, t] = (
            VmU + J_1 @ (V_T - Vm1) @ J_1.T
        )  # Equation 23 in the notes

        if t > 0:
            # Update weight
            J_2 = (
                np.squeeze(S["VmU"][:, :, t - 1])
                @ A.T
                @ np.linalg.pinv(np.squeeze(S["Vm"][:, :, t - 1]))
            )

            # Update lag 1 factor covariance matrix
            S["VmT_1"][:, :, t - 1] = (
                VmU @ J_2.T + J_1 @ (V_T1 - A @ VmU) @ J_2.T
            )  # key shortcut to getting WE adjustments

    return S
