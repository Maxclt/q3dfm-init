# flake8: noqa

import contextlib
import numpy as np
import pandas as pd
from typing import List, Dict, Union

from q3dfm.em_converged import em_converged
from q3dfm.EMstep import EMstep
from q3dfm.EMstepAR import EMstepAR
from q3dfm.runKF import runKF
from q3dfm.get_HJ import get_HJ


def dfm(
    X: np.ndarray,
    X_pred: np.ndarray,
    m: int,
    p: int,
    frq: List[str],
    isdiff: List[bool],
    *args,
) -> Dict[str, Union[int, np.ndarray]]:
    """Runs the dynamic factor model (DFM)

    Args:
        X (np.ndarray): Transformed input data (log differenced where needed)
            with outliers dropped for estimation.
        X_pred (np.ndarray): Transformed input data including outliers for
            prediction.
        m (int): Number of factors.
        p (int): Number of lags.
        frq (List[str]): Frequency of data, one of 'd', 'w', 'b', 'm', 'q', 'y'.
        isdiff (List[bool]): Logical list indicating if series is differenced.
        *args: Optional arguments:
            blocks (np.ndarray): Zeros restrictions on loadings.
            ar_errors (bool): Include AR(1) errors of the data
            threshold (float): Threshold for likelihood function convergence.
            varnames (list): Variable names for printing results.

    Returns:
        Dict[str, Union[int, np.ndarray]]: A dictionary with the results of
            the model, containing keys for fitted values, factors, loadings,
            etc.
    """

    # Initialize
    print("Estimating the dynamic factor model (DFM)...\n")

    T, k = X.shape
    isdiff = np.array(isdiff, dtype=bool)

    # Process optional arguments
    if not args:
        blocks = np.ones((k, m))
        threshold = 1e-5
        ar_errors = False
        varnames = list(range(1, k + 1))
    elif len(args) == 1:
        blocks = args[0]
        threshold = 1e-5
        ar_errors = False
        varnames = list(range(1, k + 1))
    elif len(args) == 2:
        blocks = args[0]
        ar_errors = args[1]
        threshold = 1e-5
        varnames = list(range(1, k + 1))
    elif len(args) == 3:
        blocks = args[0]
        ar_errors = args[1]
        threshold = args[2]
        varnames = list(range(1, k + 1))
    else:
        blocks = args[0]
        ar_errors = args[1]
        threshold = args[2]
        varnames = args[3]

    varnames = pd.DataFrame(varnames, columns=["Series"])

    # Check consistency of blocks matrix dimensions
    if blocks.shape[1] != m:
        raise ValueError(
            "Size of blocks is not consistent with the number of factors"
        )
    if blocks.shape[0] != k:
        raise ValueError(
            "Size of blocks is not consistent with the number of series"
        )
    if np.any(np.sum(blocks, axis=1) == 1) and ar_errors:
        raise ValueError(
            "With AR errors, blocks must allow factors to load on more than one series"
        )

    # Prepare Data
    Mx = np.nanmean(X, axis=0)
    Wx = np.nanstd(X, axis=0) / 10
    xNaN = (X - Mx) / Wx  # Standardize

    frq = set_frequencies(frq)  # Set frequencies
    A_new, H_new, Q_new, R_new = InitCond(xNaN, m, p, frq, isdiff, blocks)

    # EM Loop
    previous_loglik = -np.inf
    num_iter = 0
    converged = False
    max_iter = 1000

    Y = xNaN.T  # Transpose for faster column-wise access

    while num_iter < max_iter and not converged:
        H, R, A, Q = H_new, R_new, A_new, Q_new
        H_new, R_new, A_new, Q_new, loglik = EMstep(
            Y, A, H, Q, R, p, frq, isdiff, blocks
        )

        if num_iter > 2:
            converged, _ = em_converged(
                loglik, previous_loglik, threshold, True
            )

        if num_iter % 10 == 0 and num_iter > 0:
            print(f"Iteration {num_iter} of max {max_iter}")
            print(
                f"  Loglik: {loglik}   ({(loglik - previous_loglik) / previous_loglik * 100:.2f}%)"
            )

        eA = np.abs(np.linalg.eigvals(A_new))

        if np.max(eA) >= 1:
            print(
                "Estimated transition matrix non-stationary, breaking EM iterations"
            )
            break

        previous_loglik = loglik
        num_iter += 1

    # Handle AR errors if necessary
    if ar_errors:
        print("Estimating model with AR errors...")
        A_new, H_new, Q_new, R_new = InitAR(Y, A, H, Q, R, frq, isdiff)
        previous_loglik, num_iter, converged = -np.inf, 0, False

        while num_iter < max_iter and not converged:
            H, R, A, Q = H_new, R_new, A_new, Q_new
            H_new, R_new, A_new, Q_new, loglik = EMstepAR(
                Y, A, H, Q, R, p, frq, isdiff, blocks
            )

            if num_iter > 2:
                converged, _ = em_converged(
                    loglik, previous_loglik, threshold, True
                )

            if num_iter % 10 == 0 and num_iter > 0:
                print(f"Iteration {num_iter} of max {max_iter}")
                print(
                    f"  Loglik: {loglik} ({(loglik - previous_loglik) / previous_loglik * 100:.2f}%)"
                )

            eA = np.abs(np.linalg.eigvals(A_new))

            if np.max(eA) >= 1:
                print(
                    "Estimated transition matrix non-stationary, breaking EM iterations"
                )
                break

            previous_loglik = loglik
            num_iter += 1

    # Final Kalman Filter step
    HJ = get_HJ(H, frq, isdiff, p)
    xpNaN = (X_pred - Mx) / Wx
    if ar_errors:
        HH = np.hstack([HJ, np.eye(k)])
        Zsmooth, Vsmooth, _, LogLik, Update = runKF(xpNaN.T, A, HH, Q, R)
    else:
        Zsmooth, Vsmooth, _, LogLik, Update = runKF(xpNaN.T, A, HJ, Q, R)

    Zsmooth = Zsmooth[:, 1:].T
    Vsmooth = Vsmooth[:, :, 1:]
    var_Y = np.zeros((X_pred.shape[0], k))

    if ar_errors:
        sA = A.shape[0]
        sa = sA - k
        for t in range(X_pred.shape[0]):
            var_Y[t, :] = np.diag(HH @ Vsmooth[:, :, t] @ HH.T)
        y_common = Zsmooth[:, :sa] @ HH[:, :sa].T
        y_ar = Zsmooth[:, sa:]
        y_smooth = y_common + y_ar
    else:
        for t in range(X_pred.shape[0]):
            var_Y[t, :] = np.diag(HJ @ Vsmooth[:, :, t] @ HJ.T) + R
        y_smooth = Zsmooth @ HJ.T

    y_upper = y_smooth + np.sqrt(var_Y)
    y_lower = y_smooth - np.sqrt(var_Y)

    # Load results into a dictionary
    Res = {
        "y_smooth": y_smooth,
        "Y_smooth": Wx * y_smooth + Mx,
        "Y_upper": Wx * y_upper + Mx,
        "Y_lower": Wx * y_lower + Mx,
        "Z": Zsmooth,
        "H": H,
        "HJ": HJ,
        "R": R,
        "A": A,
        "Q": Q,
        "Mx": Mx,
        "Wx": Wx,
        "m": m,
        "p": p,
        "forecast_loglikelihood": np.real(LogLik),
        "fitted_loglikelihood": np.real(loglik),
        "Update": Update,
    }

    # Display output (Factor loadings and observation variance tables)
    with contextlib.suppress(Exception):
        print("Table 1: Factor Loadings")
        print(pd.concat([varnames, pd.DataFrame(Res["H"])], axis=1))
    with contextlib.suppress(Exception):
        print("Table 2: Observation Variance")
        print(pd.concat([varnames, pd.DataFrame(Res["R"])], axis=1))

    return Res
