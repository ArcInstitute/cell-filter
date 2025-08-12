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
    likelihoods[: matrix.indptr.size - 1] -= np.add.reduceat(
        summation_terms, matrix.indptr[:-1]
    )

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
    for s_idx in tqdm(
        np.arange(n_iter), total=n_iter, desc="Sampling draws from Multinomials..."
    ):
        start_idx = s_idx * n
        choices[start_idx : start_idx + n] = rng.choice(
            a=p_mat.shape[1], p=p_mat[s_idx], size=n
        )
    return choices


def _evaluate_simulations(
    max_total: int, n_iter: int, alpha: float, probs: np.ndarray, seed: int
) -> np.ndarray:
    # Ensure the max total is a discrete integer
    max_total = int(max_total)

    # Set the random seed
    rng = np.random.default_rng(seed=seed)

    # Sample the probabilities from a dirichlet
    logger.info("Sampling priors from Dirichlet...")
    p_hat = rng.dirichlet(alpha * probs, size=n_iter)

    # Vectorized categorical sampling for all iterations and draw sizes at once
    choices = _vectorized_categorical_sampling(
        max_total,
        p_hat,
        n_iter,
        rng,
    )

    # Reshape the choice vector into a 2D matrix
    choice_matrix = choices.reshape(max_total, n_iter)

    # Initialize the incremental count matrix
    z_matrix = np.zeros((n_iter, probs.size))

    # Initialize the sample indices for quick lookups
    sample_index = np.arange(n_iter)

    # Precompute the $\alpha p_g$ value
    ap = alpha * probs

    # Initialize the log likelihood matrix
    llik = np.zeros((max_total, n_iter))
    for c_idx in tqdm(
        np.arange(max_total), total=max_total, desc="Evaluating log likelihood..."
    ):
        # Set the multinomial draw size
        ni = c_idx + 1

        # Determine the draw identity for each iteration
        choice_at_n = choice_matrix[c_idx]

        # Isolate the draw and its associated count for each iteration
        zki = z_matrix[sample_index, choice_at_n]
        zki += 1

        # Calculate the partial likelikelihood for the multinomial
        llik[c_idx] = (
            np.log(ni)
            - np.log(ni + alpha - 1)
            + np.log(zki + ap[choice_at_n] - 1)
            - np.log(zki)
        )

    # Cumulative sum over each iteration for draw-specific likelihoods
    return llik.cumsum(axis=0)


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

    scores = _evaluate_simulations(
        unique_totals.max(),
        n_iter,
        alpha,
        probs,
        seed,
    )

    return {
        "probs": probs,
        "alpha": alpha,
        "scores": scores,
    }
