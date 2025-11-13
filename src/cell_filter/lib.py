import logging
from typing import Literal

import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import false_discovery_control

from cell_filter._dirichlet_multinomial import (
    estimate_alpha,
    eval_log_likelihood_dirichlet_multinomial,
    evaluate_simulations_dirichlet_multinomial,
)

from ._multinomial import (
    eval_log_likelihood_multinomial,
    evaluate_simulations_multinomial,
)
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
FDR_THRESHOLD = 0.01
# The seed for the random number generator
SEED = 42


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
    method: Literal["dirichlet", "multinomial"] = "multinomial",
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
    if np.all(cell_umi_counts < min_umi_threshold):
        logger.error(f"All barcodes have less than {min_umi_threshold} UMIs.")
        logger.warning("Returning empty anndata")
        return (
            adata[np.zeros(adata.shape[0], dtype=bool)],
            dict(),
        )

    # Identify ambient cells
    if cell_umi_counts.size < amb_ind_min:
        logger.error(
            f"Not enough barcodes to identify ambient cells. Found only {cell_umi_counts.size} barcodes. Need at least {amb_ind_min + 1} barcodes"
        )
        logger.warning(
            f"Returning simply filtered anndata (umis < {min_umi_threshold})"
        )
        mask = cell_umi_counts < min_umi_threshold
        return adata[mask], dict()

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
    # Estimate alpha only if using dirichlet method
    if method == "dirichlet":
        logger.info("Maximum likelihood estimation of alpha...")
        alpha = estimate_alpha(amb_matrix, probs)
        logger.info(f"Optimized alpha={alpha:.4f}...")
    else:
        logger.info("Using multinomial model (no alpha estimation)...")
        alpha = None

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
    candidate_mask = (cell_umi_counts < retain) & (cell_umi_counts >= reject_boundary)
    candidate_matrix = matrix[candidate_mask]
    candidate_totals = cell_umi_counts[candidate_mask]
    # Score simulations - use different method based on parameter
    if method == "dirichlet":
        assert alpha is not None, "alpha must be specified for Dirichlet method"
        assert alpha > 0, "alpha must be greater than 0"
        logger.info(
            f"Evaluating s={n_iter} simulations up to n={retain} unique totals (Dirichlet-Multinomial)"
        )
        sim_llik = evaluate_simulations_dirichlet_multinomial(
            retain, n_iter, alpha, probs, seed
        )

        logger.info(
            f"Evaluating likelihood for {candidate_totals.size} candidate barcodes"
        )
        obs_llik = eval_log_likelihood_dirichlet_multinomial(
            alpha, candidate_matrix, candidate_totals, probs
        )
    else:
        logger.info(
            f"Evaluating s={n_iter} simulations up to n={retain} unique totals (Multinomial)"
        )
        sim_llik = evaluate_simulations_multinomial(retain, n_iter, probs, seed)

        logger.info(
            f"Evaluating likelihood for {candidate_totals.size} candidate barcodes"
        )
        obs_llik = eval_log_likelihood_multinomial(
            candidate_matrix, candidate_totals, probs
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
    logger.info(
        f"Final number of filtered cells: {passing_candidates.size + auto_accepted.size}"
    )

    logger.info("Done!")
    return (adata[passing_cells], stats)
