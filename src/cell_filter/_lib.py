import sys

import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import betaln

from ._sgt import simple_good_turing

UMI_THRESHOLD = 10


def _eval_neg_log_likelihood(
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

    # Return the negative log likelihood
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
    ambient_bc_sum = np.array(amb_matrix.sum(axis=1)).flatten()

    # Convert probabilities
    probs = simple_good_turing(ambient_gene_sum)

    alpha = np.random.uniform(0.1, 100)
    likelihoods = _eval_neg_log_likelihood(
        alpha,
        amb_matrix,
        ambient_bc_sum,
        probs,
    )

    return likelihoods
