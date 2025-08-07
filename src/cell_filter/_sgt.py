import numpy as np


def _calculate_z(
    r: np.ndarray,
    n: np.ndarray,
) -> np.ndarray:
    assert r.shape == n.shape, "r and n must have the same shape"
    z = np.zeros_like(r)

    idx = 0
    while idx < r.size:
        if idx == 0:
            z[idx] = 2 * n[idx] / r[idx + 1]
        elif idx == r.size - 1:
            z[idx] = n[idx] / (r[idx] - r[idx - 1])
        else:
            z[idx] = 2 * n[idx] / (r[idx + 1] - r[idx - 1])
        idx += 1

    return z


def _calculate_best_fit(
    log_r: np.ndarray,
    log_z: np.ndarray,
) -> np.ndarray:
    assert log_r.shape == log_z.shape, "log_r and log_z must have the same shape"
    design_matrix = np.column_stack((log_r, np.ones_like(log_r)))
    return np.linalg.lstsq(design_matrix, log_z, rcond=None)[0]


def _S(coeffs: np.ndarray, r: float) -> float:
    return np.exp(coeffs[0] * np.log(r) + coeffs[1])


def _calculate_r_star(
    r: np.ndarray,
    n: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    r_star = np.zeros_like(r)

    before_break = True
    for idx in np.arange(r.size):
        # Calculate y value
        y = (r[idx] + 1) * (_S(coeffs, r[idx] + 1) / _S(coeffs, r[idx]))

        if before_break:
            # Only calculate x values before breakpoint
            x = (r[idx] + 1) * (n[idx + 1] / n[idx])

            term1 = (r[idx] + 1) ** 2
            term2 = n[idx + 1] / n[idx] ** 2
            term3 = 1 + (n[idx + 1] / n[idx])
            comp = 1.96 * np.sqrt(term1 * term2 * term3)

            if np.abs(x - y) > comp:
                r_star[idx] = x
            else:
                before_break = False
                r_star[idx] = y

        else:
            r_star[idx] = y

    return r_star


def _adjust_original_frequencies(
    frequencies: np.ndarray,
    r: np.ndarray,
    p: np.ndarray,
    p0: float,
) -> np.ndarray:
    adj = np.zeros_like(frequencies)
    r_to_p = dict(zip(r, p))
    zero_prob = p0 / np.sum(frequencies == 0)
    for idx, val in enumerate(frequencies):
        if val == 0:
            adj[idx] = zero_prob
        else:
            adj[idx] = r_to_p[val]
    return adj


def simple_good_turing(frequencies: np.ndarray):
    """
    Apply the Simple Good-Turing (SGT) method to smooth the frequency distribution of an input count vector.

    This follows the algorithm specified in the paper "Good-Turing Frequency Estimation Without Tears"

    # References
    1. William A. Gale & Geoffrey Sampson (1995) Good‐turing frequency estimation without tears, Journal of Quantitative Linguistics, 2:3, 217-237, DOI:10.1080/09296179508590051

    # Arguments:
    frequencies: np.ndarray
        A count vector, this should be a 1D array of non-negative integers (zeros can be included).

    # Returns:
    np.ndarray
        The smoothed frequency distribution (0, 1). This will sum to 1.
    """
    # Limit frequencies to positive non-zero values
    nz_frequencies = frequencies[frequencies > 0]

    # Frequency (n) of each unique frequency (r)
    r, n = np.unique(nz_frequencies, return_counts=True)

    # Total number of observations
    N = frequencies.sum()

    # Set the unseen probability
    p0 = 0
    if r[0] == 1:
        p0 = n[0] / N

    # Determine the Z vector
    z = _calculate_z(r, n)

    # calculate log
    log_r = np.log(r)
    log_z = np.log(z)

    # calculate best fit
    coeffs = _calculate_best_fit(log_r, log_z)

    # calculate r_star
    r_star = _calculate_r_star(r, n, coeffs)

    # Calculate N'
    N_dash = np.sum(n * r_star)

    # Recalculate probabilities
    p = (1 - p0) * (r_star / N_dash)

    # Adjust original frequencies
    adj = _adjust_original_frequencies(frequencies, r, p, p0)

    return adj
