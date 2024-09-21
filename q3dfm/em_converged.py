import numpy as np


def em_converged(
    loglik: float,
    previous_loglik: float,
    threshold: float = 1e-4,
    check_decreased: bool = True,
):
    """
    Checks whether the EM algorithm has converged.

    Args:
        loglik (float): Log-likelihood from the current EM iteration.
        previous_loglik (float): Log-likelihood from the previous EM iteration.
        threshold (float, optional): Convergence threshold. Defaults to 1e-4.
        check_decreased (bool, optional): If True, checks if the
            log-likelihood has decreased. Defaults to True.

    Returns:
        converged (bool): True if convergence criteria are satisfied, False
            otherwise.
        decrease (bool): True if the log-likelihood has decreased, False
            otherwise.
    """
    decrease = False

    # Check if log-likelihood decreases (optional)
    if check_decreased and loglik - previous_loglik < -1e-3:
        print(
            f"likelihood decreased from {previous_loglik:.4f} to {loglik:.4f}!"
        )
        decrease = True

    # Check convergence criteria
    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (
        abs(loglik) + abs(previous_loglik) + np.finfo(float).eps
    ) / 2  # Avoid division by zero

    converged = delta_loglik / avg_loglik < threshold
    return converged, decrease
