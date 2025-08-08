import logging

import anndata as ad
import numpy as np
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.sparse import csr_matrix
from scipy.special import betaln
from tqdm import tqdm

from ._sgt import simple_good_turing

logger = logging.getLogger(__name__)

UMI_THRESHOLD = 100
N_SIMULATIONS = 1000
SEED = 42


def _eval_log_likelihood(
    alpha: float,
    matrix: csr_matrix,
    total: np.ndarray,
    probs: np.ndarray,
):
    """Evaluate the negative log likelihood of the Dirichlet-Multinomial distribution.

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
    for idx in np.arange(matrix.indptr.size - 1):
        lb = matrix.indptr[idx]  # Pull the left-bound of the summation
        ub = matrix.indptr[idx + 1]  # Pull the right-bound of the summation
        likelihoods[idx] -= np.sum(summation_terms[lb:ub])

    # Return the log likelihood
    return likelihoods


def _estimate_alpha(matrix: csr_matrix, probs: np.ndarray):
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
        lambda alpha: -_eval_log_likelihood(alpha, matrix, bc_sum, probs).sum(),
        bounds=(1e-6, 1000),
        method="bounded",
    )
    if not result.success or not isinstance(result, OptimizeResult):
        raise ValueError("Optimization failed")
    return result.x


def _vectorized_categorical_sampling(
    n: int,
    p_mat: np.ndarray,
    n_iter: int,
    rng,
) -> np.ndarray:
    """Fills a buffer with random categories.

    This is used to simulate a non-unique categorical distribution which can
    then be counted to generate a multinomial distribution.

    It is an optimization for cases where `n` is small compared to the number of categories
    and the expected number of categories sampled is sparse.

    Layout is:
    [n=1   ] [n=2   ] [n=3   ] [...]
    [n_iter] [n_iter] [n_iter] [...]
    """
    choices = np.zeros(n_iter * n, dtype=int)
    for s_idx in np.arange(n_iter):
        start_idx = s_idx * n
        choices[start_idx : start_idx + n] = rng.choice(
            a=p_mat.shape[1], p=p_mat[s_idx], size=n
        )
    return choices


def _csr_multinomial(
    n: int,
    p_mat: np.ndarray,
    n_iter: int,
    choices: np.ndarray,
    sample_indices: np.ndarray,
):
    """Perform an efficient sampling from a multinomial distribution.

    This works well when `n << len(p)` as the expected sampled matrix is sparse.

    This function uses vectorized operations to efficiently sample from a multinomial distribution.
    It's optimized for speed and memory efficiency.

    # Inputs:
    n: int
        The number of trials for each sample
    p_mat: np.ndarray
        A 2D array of probabilities of size (n_iter, len(p))
    n_iter: int
        The number of iterations to perform
    choices: np.ndarray
        The sampled categories for each sample (1D).
        This array is arranged as follows:
            [n=1   ] [n=2   ] [n=3   ] [...]
            [n_iter] [n_iter] [n_iter] [...]

    # Outputs:
    csr_matrix
        The sampled count matrix
    """
    if n == 0:
        return csr_matrix((n_iter, p_mat.shape[1]), dtype=int)

    # Encode coordinates
    #
    # This shifts all choice categories by the category size
    # providing a unique category set for each sample
    coords = sample_indices[: n_iter * n] * p_mat.shape[1] + choices[: n_iter * n]

    # Vectorized count over all samples
    unique_coords, counts = np.unique(coords, return_counts=True)

    # Decode coordinates back to the original categories
    rows = unique_coords // p_mat.shape[1]
    cols = unique_coords % p_mat.shape[1]

    return csr_matrix((counts, (rows, cols)), shape=(n_iter, p_mat.shape[1]), dtype=int)


def _score_simulations(
    unique_totals: np.ndarray,
    probs: np.ndarray,
    alpha: float,
    n_iter: int = N_SIMULATIONS,
    seed: int = SEED,
) -> np.ndarray:
    """Score simulations for unique barcode umi totals.

    This is a performance optimization that reuses the same resampled matrices
    for a specific barcode UMI total.

    # Inputs:
    unique_totals: np.ndarray
        The unique barcode UMI totals to score
    probs: np.ndarray
        The probability of each gene being expressed
    alpha: float
        The alpha parameter of the DM distribution
    n_iter: int
        The number of simulations to run
    seed: int
        The random seed to use for the simulations

    # Outputs:
    scores: np.ndarray (unique_totals.size, n_iter)
    """
    rng = np.random.default_rng(seed=seed)

    # Convert unique_totals to integer array
    unique_totals = unique_totals.astype(int)
    logger.info(f"Performing {unique_totals.size} simulations...")

    # Sample the adjusted multinomial probabilities for all iterations at once
    logger.info("Sampling from Dirichlet...")
    p_hat = rng.dirichlet(alpha * probs, size=n_iter)

    # Precompute the categorical sampling
    logger.info("Sampling Categories...")
    choices = _vectorized_categorical_sampling(unique_totals.max(), p_hat, n_iter, rng)

    # Precompute the sample indices
    sample_indices = np.repeat(np.arange(n_iter), unique_totals.max())

    # For each unique total sample all iterations at once and score
    scores = np.zeros((unique_totals.size, n_iter))
    for idx, total in tqdm(
        enumerate(unique_totals), desc="Scoring simulations", total=unique_totals.size
    ):
        matrices = _csr_multinomial(total, p_hat, n_iter, choices, sample_indices)
        scores[idx] = _eval_log_likelihood(
            alpha, matrices, np.repeat(total, n_iter), probs
        )

    return scores


def empty_drops(
    adata: ad.AnnData,
    threshold: float | int = UMI_THRESHOLD,
    n_iter: int = N_SIMULATIONS,
    seed: int = SEED,
):
    logging.basicConfig(level=logging.INFO)
    """Empty drops detection with optional multiprocessing."""
    if not isinstance(adata.X, csr_matrix):
        logger.info("Converting data to csr_matrix...")
        adata.X = csr_matrix(adata.X)
        logger.info("Finished converting data to csr_matrix.")

    # Extract matrix from AnnData object
    matrix = adata.X

    if threshold <= 0:
        logger.error("threshold must be positive non-zero")
        raise ValueError("threshold must be positive non-zero")

    # Determine cell UMI counts
    logger.info("Determining cell UMI counts...")
    cell_umi_counts = np.array(matrix.sum(axis=1)).flatten()

    # Identify ambient cells
    logger.info("Identifying ambient cells...")
    ambient_mask = cell_umi_counts < threshold

    # Extract ambient matrix
    logger.info("Extracting ambient matrix...")
    amb_matrix = matrix[ambient_mask]

    logger.info("Calculating ambient gene sum...")
    ambient_gene_sum = np.array(amb_matrix.sum(axis=0)).flatten()

    # Convert probabilities
    logger.info("Converting probabilities (SGT)...")
    probs = simple_good_turing(ambient_gene_sum)

    # Estimate alpha
    logger.info("Maximum likelihood estimation of alpha...")
    alpha = _estimate_alpha(amb_matrix, probs)

    # Score simulations (now with multiprocessing)
    unique_totals = np.unique(cell_umi_counts)
    logger.info(f"Scoring simulations for {len(unique_totals)} unique totals")

    scores = _score_simulations(
        unique_totals,
        probs,
        alpha,
        n_iter=n_iter,
        seed=seed,
    )

    return {"probs": probs, "alpha": alpha, "scores": scores}
