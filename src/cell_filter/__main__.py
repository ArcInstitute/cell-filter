import typer
import anndata as ad
import logging
from cell_filter.lib import (
    MIN_UMI_THRESHOLD,
    N_EXPECTED_CELLS,
    MAX_PERCENTILE,
    MAX_MIN_RATIO,
    UMI_MIN_FRAC,
    AMB_INDEX_MIN,
    AMB_INDEX_MAX,
    N_SIMULATIONS,
    FDR_THRESHOLD,
    SEED,
    empty_drops,
)


def main(
    path_h5ad: str,
    path_filt: str,
    min_umi_threshold: int = MIN_UMI_THRESHOLD,
    n_expected_cells: int = N_EXPECTED_CELLS,
    max_percentile: float = MAX_PERCENTILE,
    max_min_ratio: float = MAX_MIN_RATIO,
    umi_min_frac: float = UMI_MIN_FRAC,
    amb_ind_min: int = AMB_INDEX_MIN,
    amb_ind_max: int = AMB_INDEX_MAX,
    n_iter: int = N_SIMULATIONS,
    fdr_threshold: float = FDR_THRESHOLD,
    seed: int = SEED,
    verbose: bool = False,
    logfile: str | None = None,
):
    logger = logging.getLogger("cell-filter")
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    if verbose or logfile:
        logger.setLevel(logging.INFO)
        if logfile:
            logger.addHandler(logging.FileHandler(logfile))
        logger.info("Verbose mode enabled")
    else:
        logger.setLevel(logging.WARNING)

    logger.info(f"Reading in h5ad from {path_h5ad}...")
    adata = ad.read_h5ad(path_h5ad)

    filt, _stats = empty_drops(
        adata,
        min_umi_threshold=min_umi_threshold,
        n_expected_cells=n_expected_cells,
        max_percentile=max_percentile,
        max_min_ratio=max_min_ratio,
        umi_min_frac=umi_min_frac,
        amb_ind_min=amb_ind_min,
        amb_ind_max=amb_ind_max,
        n_iter=n_iter,
        fdr_threshold=fdr_threshold,
        seed=seed,
        verbose=verbose,
        logfile=logfile,
    )

    logger.info(f"Writing filtered h5ad to {path_filt}")
    filt.write_h5ad(path_filt)


def app():
    typer.run(main)
