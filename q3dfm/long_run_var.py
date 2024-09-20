import numpy as np


def long_run_var(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Calculate the long run variance of the transition equation given the
    paremeters A and Q

    Args:
        A (np.ndarray): r-by-r transition matrix
        Q (np.ndarray): r-by-r covariance matrix for transition equation
            residuals

    Returns:
        np.ndarray: r-by-r long run variance matrix
    """
    sA = A.shape[0]  # size of the transition matrix (square)
    xx = np.eye(sA**2) - np.kron(
        A, A
    )  # Create the Kronecker product of A with itself and subtract it from
    # the identity matrix
    vQ = Q.reshape(sA**2, 1)  # Vectorize Q (flatten it into a column vector)
    V_0 = np.linalg.solve(
        xx, vQ
    )  # Solve the system of linear equations xx * V_0 = vQ
    V_0 = V_0.reshape(
        sA, sA
    )  # Reshape V_0 back into the original matrix shape (sA-by-sA)
    V_0 = (V_0 + V_0.T) / 2  # Reduce rounding error by symmetrizing the matrix
    return V_0
