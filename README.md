# cell-filter

A python implementation of the [EmptyDrops](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1662-y) algorithm.

This is an implementation following the CellRanger/STARSolo algorithm which only evaluates candidate droplets within a specific band of the UMI total distribution (under the knee until a minimum UMI threshold).

## Installation

`cell-filter` is distributed using [uv](https://docs.astral.sh/uv/)

```bash
# Install cell-filter
uv tool install cell-filter

# Check installation
cell-filter --help
```

## Usage

To run `cell-filter` with default parameters you'll need two arguments:

1. The path to the input h5ad file containing **unfiltered** data.
2. The path to the write the output h5ad file containing **filtered** data.

```bash
cell-filter <input.h5ad> <output.h5ad>
```

Feel free to explore the code and adjust the parameters to suit your needs.
