import numba as nb
import numpy as np
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.sparse import csr_matrix
from scipy.special import betaln


@nb.njit()
def _fill_categories(
    buffer: np.ndarray,
    categories: np.ndarray,
    rand_buffer: np.ndarray,
    n: int,
    p: np.ndarray,
):
    assert buffer.size == n, f"buffer size {buffer.size} != n {n}"
    assert rand_buffer.size == n, f"rand_buffer size {rand_buffer.size} != n {n}"
    assert categories.size == p.size, (
        f"categories size {categories.size} != p size {p.size}"
    )
    rand_buffer[:] = np.random.random(size=rand_buffer.size)
    buffer[:] = categories[
        np.searchsorted(
            np.cumsum(p),
            rand_buffer,
            side="right",
        )
    ]


@nb.njit()
def _fill_llik_dirichlet_multinomial(
    llik: np.ndarray,
    z_buffer: np.ndarray,
    p_buffer: np.ndarray,
    c_buffer: np.ndarray,
    categories: np.ndarray,
    r_buffer: np.ndarray,
    max_total: int,
    alpha: float,
    probs: np.ndarray,
    n_iter: int,
    seed: int,
):
    np.random.seed(seed)
    ap = alpha * probs
    for s_idx in np.arange(n_iter, dtype=np.int64):
        # Clear the z_buffer
        z_buffer[:] = 0

        # Draw from dirichlet
        p_buffer[:] = np.random.dirichlet(ap)

        # Draw all categories for iteration group at once
        _fill_categories(c_buffer, categories, r_buffer, max_total, p_buffer)

        for n_idx in np.arange(max_total):
            # set the multinomial draw size
            ni = n_idx + 1

            # Determine the draw identity for the iteration
            choice_at_n = c_buffer[n_idx]

            # Isolate the draw count and increment it
            z_buffer[choice_at_n] += 1
            zki = z_buffer[choice_at_n]

            # Compute the partial log-likelihood for the multinomial
            llik[n_idx, s_idx] = (
                np.log(ni)
                - np.log(ni + alpha - 1)
                + np.log(zki + ap[choice_at_n] - 1)
                - np.log(zki)
            )


def evaluate_simulations_dirichlet_multinomial(
    max_total: int, n_iter: int, alpha: float, probs: np.ndarray, seed: int
) -> np.ndarray:
    # Ensure the max total is a discrete integer
    max_total = int(max_total)

    # Reusable buffers
    p_buffer = np.zeros(probs.size)  # probabilities
    c_buffer = np.zeros(max_total, dtype=int)  # categories
    r_buffer = np.zeros(max_total)  # random numbers
    z_buffer = np.zeros(probs.size)  # incremental counts

    # Used for random sampling of categories
    categories = np.arange(probs.size, dtype=int)

    # Log-Likelihoods
    llik = np.zeros((max_total, n_iter))

    _fill_llik_dirichlet_multinomial(
        llik,
        z_buffer,
        p_buffer,
        c_buffer,
        categories,
        r_buffer,
        max_total,
        alpha,
        probs,
        n_iter,
        seed,
    )

    # Calculate the cumulative sum inplace
    np.cumsum(llik, axis=0, out=llik)

    return llik


def eval_log_likelihood_dirichlet_multinomial(
    alpha: float,
    matrix: csr_matrix,
    total: np.ndarray,
    probs: np.ndarray,
):
    """Evaluate the log likelihood of the Dirichlet-Multinomial distribution.

    Uses an efficient vectorized implementation.

    # Arguments
    alpha: float
        The scaling factor for the Dirichlet prior
    matrix: csr_matrix
        The observed counts for each gene across all barcodes `b`
    total: np.ndarray
        The total number of transcripts across all barcodes `b`
    probs: np.ndarray
        The probability of each gene being expressed
    """
    # Scale the gene probabilities
    alpha_g = alpha * probs

    # Calculate bc-constant term before loop
    likelihoods = np.log(total) + betaln(total, alpha)

    # Calculate the vectorized summation term
    summation_terms = np.log(matrix.data) + betaln(matrix.data, alpha_g[matrix.indices])

    # Update the likelihood inplace
    likelihoods[: matrix.indptr.size - 1] -= np.add.reduceat(
        summation_terms, matrix.indptr[:-1]
    )

    # Return the log likelihood
    return likelihoods


def estimate_alpha(matrix: csr_matrix, probs: np.ndarray):
    """Estimate the alpha parameter by optimizing the maximum likelihood of the DM distribution.

    # Inputs:
    matrix: csr_matrix
        The count matrix of shape (n_cells, n_genes)
    probs: np.ndarray
        The probability of each gene being expressed
    """
    bc_sum = np.array(matrix.sum(axis=1)).flatten()

    # Optimize alpha
    result = minimize_scalar(
        lambda alpha: -eval_log_likelihood_dirichlet_multinomial(
            alpha, matrix, bc_sum, probs
        ).sum(),
        bounds=(1e-6, 10000),
        method="bounded",
    )
    if not result.success or not isinstance(result, OptimizeResult):  # type: ignore
        raise ValueError("Optimization failed")
    return result.x
