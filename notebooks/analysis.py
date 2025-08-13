import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell
def _():
    import anndata as ad
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.sparse import csr_matrix, csc_matrix
    from pyinstrument import Profiler
    return (ad,)


@app.cell
def _(ad):
    adata = ad.read_h5ad("./data/BC001.h5ad")
    return (adata,)


@app.cell
def _(adata):
    from cell_filter import empty_drops

    filtered, result = empty_drops(adata, n_iter=50000)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
