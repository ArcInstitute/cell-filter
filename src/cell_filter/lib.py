import logging

import anndata as ad
import numpy as np
import numba as nb
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.sparse import csr_matrix
from scipy.special import betaln
from scipy.stats import false_discovery_control

from ._sgt import simple_good_turing

# The minimum UMI threshold to immediately discard barcodes
MIN_UMI_THRESHOLD = 500
# The number of expected cells in the dataset
N_EXPECTED_CELLS = 20000
# Considers the 95th percentile cell as the highest UMI total
MAX_PERCENTILE = 0.95
# The ratio with which to set the immediate acceptance threshold (the candidate upper bound)
MAX_MIN_RATIO = 5
# The fraction of the median accepted cells to set the candidate lower bound
UMI_MIN_FRAC = 0.01
# The index lower bound for ambient barcodes (descending sort of barcode UMI sums)
AMB_INDEX_MIN = 45000
# The index upper bound for ambient barcodes (descending sort of barcode UMI sums)
AMB_INDEX_MAX = 90000
# The number of simulations to perform
N_SIMULATIONS = 10000
# The threshold for the false discovery rate
FDR_THRESHOLD = 0.1
# The seed for the random number generator
SEED = 42


def _eval_log_likelihood(
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
        bounds=(1e-6, 10000),
        method="bounded",
    )
    if not result.success or not isinstance(result, OptimizeResult):
        raise ValueError("Optimization failed")
    return result.x


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
def _fill_llik(
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


def _evaluate_simulations(
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

    _fill_llik(
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


def _evaluate_pvalue(
    obs: float,
    background: np.ndarray,
) -> float:
    r = np.sum(background <= obs)
    return float((r + 1) / (background.size + 1))


def _score_candidate_barcodes(
    obs_llik: np.ndarray,
    obs_totals: np.ndarray,
    sim_llik: np.ndarray,
) -> np.ndarray:
    pvalues = np.zeros(obs_totals.size)
    for idx in np.arange(pvalues.size):
        pvalues[idx] = _evaluate_pvalue(
            obs_llik[idx],
            sim_llik[obs_totals[idx]],
        )
    return false_discovery_control(pvalues, method="bh")


def empty_drops(
    adata: ad.AnnData,
    min_umi_threshold: int = MIN_UMI_THRESHOLD,
    n_expected_cells: int = N_EXPECTED_CELLS,
    max_percentile: float = MAX_PERCENTILE,
    max_min_ratio: float | int = MAX_MIN_RATIO,
    umi_min_frac: float = UMI_MIN_FRAC,
    amb_ind_min: int = AMB_INDEX_MIN,
    amb_ind_max: int = AMB_INDEX_MAX,
    n_iter: int = N_SIMULATIONS,
    fdr_threshold: float = FDR_THRESHOLD,
    seed: int = SEED,
    verbose: bool = False,
    logfile: str | None = None,
) -> tuple[ad.AnnData, dict]:
    # Enforce typing on inputs
    min_umi_threshold = int(min_umi_threshold)
    n_expected_cells = int(n_expected_cells)
    max_percentile = float(max_percentile)
    max_min_ratio = int(max_min_ratio)
    umi_min_frac = float(umi_min_frac)
    amb_ind_min = int(amb_ind_min)
    amb_ind_max = min(int(amb_ind_max), adata.shape[0])
    n_iter = int(n_iter)
    fdr_threshold = float(fdr_threshold)
    seed = int(seed)
    """Empty drops filtering"""
    logger = logging.getLogger("cell-filter")
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    if verbose or logfile:
        logger.setLevel(logging.INFO)
        if logfile:
            logger.addHandler(logging.FileHandler(logfile))
    else:
        logger.setLevel(logging.WARNING)

    if not isinstance(adata.X, csr_matrix):
        logger.info("Converting data to csr_matrix...")
        adata.X = csr_matrix(adata.X)
        logger.info("Finished converting data to csr_matrix.")

    # Extract matrix from AnnData object
    matrix = adata.X
    logger.info(f"Processing matrix size: {matrix.shape}")

    if min_umi_threshold <= 0:
        logger.error("threshold must be positive non-zero")
        raise ValueError("threshold must be positive non-zero")

    # Determine cell UMI counts
    logger.info("Determining cell UMI counts...")
    cell_umi_counts = np.array(matrix.sum(axis=1)).flatten().astype(int)

    # Identify ambient cells
    logger.info(f"Identifying {amb_ind_max - amb_ind_min} ambient cells...")
    argsorted_cell_umi_counts = np.argsort(cell_umi_counts)[::-1]  # descending order
    ambient_mask = argsorted_cell_umi_counts[amb_ind_min:amb_ind_max]

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
    logger.info(f"Optimized alpha={alpha:.4f}...")

    # Identify the retainment boundary
    max_ind = int(np.round(n_expected_cells * (1.0 - max_percentile)))
    n_umi_max = int(cell_umi_counts[argsorted_cell_umi_counts[max_ind]])
    retain = int(max(n_umi_max / max_min_ratio, 1))
    n_valid = int(np.sum(cell_umi_counts >= retain))
    logger.info(f"Retainment boundary: {retain} UMIs ({n_valid} auto-accepted cells)")

    # Identify the auto-reject boundary
    median_idx = min(n_valid // 2, len(cell_umi_counts) - 1)
    median_umi = cell_umi_counts[argsorted_cell_umi_counts[median_idx]]
    reject_boundary = max(
        min_umi_threshold,
        int(np.round(umi_min_frac * median_umi)),
    )
    logger.info(f"Rejection boundary: {reject_boundary} UMIs")

    # Score simulations (now with multiprocessing)
    logger.info(f"Evaluating s={n_iter} simulations up to n={retain} unique totals")
    sim_llik = _evaluate_simulations(
        retain,
        n_iter,
        alpha,
        probs,
        seed,
    )

    # Score the likelihood of the candidate barcodes
    candidate_mask = (cell_umi_counts < retain) & (cell_umi_counts >= reject_boundary)
    candidate_matrix = matrix[candidate_mask]
    candidate_totals = cell_umi_counts[candidate_mask]
    logger.info(f"Evaluating likelihood for {candidate_totals.size} candidate barcodes")
    obs_llik = _eval_log_likelihood(
        alpha,
        candidate_matrix,
        candidate_totals,
        probs,
    )

    # candidate false-discovery-rates
    logger.info(f"Evaluating pvalues for {candidate_totals.size} candidate barcodes")
    fdr = _score_candidate_barcodes(
        obs_llik,
        candidate_totals,
        sim_llik,
    )

    # build the mask of fully passing cells
    passing_candidates = np.flatnonzero(fdr < fdr_threshold)
    passing_candidates_in_original_index = np.flatnonzero(candidate_mask)[
        passing_candidates
    ]
    auto_accepted = np.flatnonzero(cell_umi_counts >= retain)
    passing_cells = np.unique(
        np.concatenate([auto_accepted, passing_candidates_in_original_index])
    )
    logger.info(
        f"Identified {passing_candidates.size} passing candidates and {auto_accepted.size} retained cells."
    )
    stats = {
        "probs": probs,
        "alpha": alpha,
        "sim_llik": sim_llik,
        "obs_llik": obs_llik,
        "obs_totals": candidate_totals,
        "fdr": fdr,
        "n_iter": n_iter,
    }

    logger.info("Done!")
    return (adata[passing_cells], stats)
