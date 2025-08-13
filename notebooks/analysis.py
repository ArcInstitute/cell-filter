import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell
def _():
    import anndata as ad
    import numpy as np

    return (ad,)


@app.cell
def _(ad):
    # Load in AnnData
    adata = ad.read_h5ad("./data/BC001.h5ad")
    return (adata,)


@app.cell
def _(adata):
    from cell_filter import empty_drops

    # Filter AnnData
    filtered, _stats = empty_drops(
        adata,
        n_iter=10000,
    )
    return


if __name__ == "__main__":
    app.run()
