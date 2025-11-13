import numba as nb
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import gammaln


@nb.njit()
def _fill_llik_multinomial(
    llik: np.ndarray,
    z_buffer: np.ndarray,
    c_buffer: np.ndarray,
    categories: np.ndarray,
    r_buffer: np.ndarray,
    max_total: int,
    probs: np.ndarray,
    n_iter: int,
    seed: int,
):
    """Multinomial version - follows same pattern as _fill_llik but simpler."""
    np.random.seed(seed)
    logp = np.log(probs)
    p_cumulative = np.cumsum(probs)
    p_cumulative[-1] = 1.0  # enforce 1.0 despite numerical instability

    for s_idx in np.arange(n_iter, dtype=np.int64):
        # Clear the z_buffer
        z_buffer[:] = 0

        # Draw samples directly from multinomial (no Dirichlet step)
        r_buffer[:] = np.random.random(size=max_total)
        c_buffer[:] = categories[np.searchsorted(p_cumulative, r_buffer, side="right")]

        for n_idx in np.arange(max_total):
            ni = n_idx + 1
            choice_at_n = c_buffer[n_idx]

            # Increment count (same as your DM version!)
            z_buffer[choice_at_n] += 1
            zki = z_buffer[choice_at_n]

            # Multinomial incremental log-likelihood (simpler than DM)
            llik[n_idx, s_idx] = np.log(ni) - np.log(zki) + logp[choice_at_n]


def evaluate_simulations_multinomial(
    max_total: int, n_iter: int, probs: np.ndarray, seed: int
) -> np.ndarray:
    """Multinomial version - same structure as _evaluate_simulations."""
    max_total = int(max_total)

    # Reusable buffers (same as DM version)
    c_buffer = np.zeros(max_total, dtype=int)
    r_buffer = np.zeros(max_total)
    z_buffer = np.zeros(probs.size)
    categories = np.arange(probs.size, dtype=int)

    # Log-Likelihoods
    llik = np.zeros((max_total, n_iter))

    _fill_llik_multinomial(
        llik,
        z_buffer,
        c_buffer,
        categories,
        r_buffer,
        max_total,
        probs,
        n_iter,
        seed,
    )

    # Calculate cumulative sum (same as DM version)
    np.cumsum(llik, axis=0, out=llik)

    return llik


def eval_log_likelihood_multinomial(
    matrix: csr_matrix,
    total: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    """Evaluate multinomial log-likelihood for observed data.

    Args:
        matrix: Observed counts (features x barcodes)
        total: Total UMIs per barcode
        probs: Feature probabilities

    Returns:
        Log-likelihoods for each barcode
    """
    logp = np.log(probs)
    num_bcs = matrix.shape[0]  # type: ignore
    loglk = np.zeros(num_bcs, dtype=float)

    consts = gammaln(total + 1)

    for i in range(num_bcs):
        idx_start, idx_end = matrix.indptr[i], matrix.indptr[i + 1]
        idxs = matrix.indices[idx_start:idx_end]
        row = matrix.data[idx_start:idx_end]
        short_logp = logp[idxs]
        loglk[i] = consts[i] - gammaln(row + 1).sum() + (row * short_logp).sum()

    return loglk
