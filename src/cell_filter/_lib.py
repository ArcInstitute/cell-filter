import anndata as ad
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.special import betaln

from ._sgt import simple_good_turing

UMI_THRESHOLD = 10


def _log_dm_lik(
    alpha: float,
    total: float,
    counts: np.ndarray,
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
    if isinstance(counts, csr_matrix) or isinstance(counts, csc_matrix):
        summation = np.sum(
            np.log(counts.data) + betaln(counts.data, alpha_g[: counts.data.size])  # type: ignore
        )
    else:
        summation = np.sum(np.log(counts) + betaln(counts, alpha_g))

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
    if threshold <= 0:
        raise ValueError("threshold must be positive non-zero")

    cell_umi_counts = adata.X.sum(axis=1)  # type: ignore
    ambient_mask = cell_umi_counts < threshold

    ambient_adata = adata[ambient_mask]
    ambient_gene_sum = np.array(
        ambient_adata.X.sum(axis=0)  # type: ignore
    ).flatten()

    probs = simple_good_turing(ambient_gene_sum)

    alpha = np.random.uniform(0.1, 100)
    likelihoods = _eval_neg_log_likelihood(
        alpha,
        csr_matrix(ambient_adata.X),  # type: ignore
        probs,
    )
