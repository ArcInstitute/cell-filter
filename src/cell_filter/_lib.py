import sys

import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import betaln

from ._sgt import simple_good_turing

UMI_THRESHOLD = 10


def _log_dm_lik(
    alpha: float,
    total: float,
    counts: csr_matrix,
    probs: np.ndarray,
):
    """Log of the Dirichlet-Multinomial Likelihood.

    # Arguments
    alpha: float
        The scaling factor for the Dirichlet prior
    total: float
        The total number of transcripts for barcode `b`
    counts: np.ndarray
        The observed counts for each gene in barcode `b`
    probs: np.ndarray
        The probability of each gene being expressed
    """
    # Constant term
    constant = np.log(total) + betaln(total, alpha)
    alpha_g = alpha * probs

    # Summation term
    summation = np.sum(
        np.log(counts.data) + betaln(counts.data, alpha_g[counts.indices])
    )

    return constant - summation


def _eval_neg_log_likelihood(
    alpha: float,
    matrix: np.ndarray | csr_matrix,
    probs: np.ndarray,
):
    """Evaluate the negative log likelihood of the Dirichlet-Multinomial distribution.

    # Arguments
    alpha: float
        The scaling factor for the Dirichlet prior
    matrix: np.ndarray | csr_matrix
        The observed counts for each gene in barcode `b`
    probs: np.ndarray
        The probability of each gene being expressed
    """
    total = np.array(matrix.sum(axis=1)).flatten()
    likelihoods = np.zeros_like(total)
    for idx in np.arange(total.size):
        likelihoods[idx] = _log_dm_lik(alpha, total[idx], matrix[idx], probs)
    return -likelihoods.sum()


def empty_drops(
    adata: ad.AnnData,
    threshold: float | int = UMI_THRESHOLD,
):
    if not isinstance(adata.X, csr_matrix):
        print("Converting data to csr_matrix...", file=sys.stderr)
        adata.X = csr_matrix(adata.X)
        print("Finished converting data to csr_matrix.", file=sys.stderr)

    # Extract matrix from AnnData object
    matrix = adata.X

    if threshold <= 0:
        raise ValueError("threshold must be positive non-zero")

    # Determine cell UMI counts
    cell_umi_counts = np.array(matrix.sum(axis=1)).flatten()

    # Identify ambient cells
    ambient_mask = cell_umi_counts < threshold

    # Extract ambient matrix
    amb_matrix = matrix[ambient_mask]
    ambient_gene_sum = np.array(amb_matrix.sum(axis=0)).flatten()

    # Convert probabilities
    probs = simple_good_turing(ambient_gene_sum)

    alpha = np.random.uniform(0.1, 100)
    likelihoods = _eval_neg_log_likelihood(
        alpha,
        amb_matrix,
        probs,
    )

    return likelihoods
