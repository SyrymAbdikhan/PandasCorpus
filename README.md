# PandasCorpus: A Dataset of Pandas Workflows on GitHub

## Overview

PandasCorpus is a dataset of Jupyter notebooks containing the pandas library crawled from GitHub. This version of the dataset was collected in August 2025, downloading 140k notebooks (18 GB of data). All the source code and the datasets are available.

## Contents

The PandasCorpus contains the following dataset files (CSV or Parquet versions)

Dataset:

- **code_cells.csv (or .parquet.gzip)**: 3.8 million cells of code cells with fields: notebook_hash, cell_id, and source
- **markdown_cells.csv (or .parquet.gzip)**: 1.4 million cells of markdown cells with the same fields as code_cells.csv

Metrics and links:

- **code_cells_metrics.csv**: the metrics gathered from every code cell
- **pandas_ops_count.csv**: count of pandas operations over all cells
- **code_links.csv**: links to repositories and files of downloaded notebooks
- **link_hash_map.csv**: lookup table connecting code cells with code links

API Operations and others:

- **csv_count.csv**: number of csv files in each repository
- **csv_links.csv (or .parquet.gzip)**: links to csv files for each repository
- **pandas_ops.csv**: dataframe and general functions that output dataframe, series, or array-like types
- **mpl_ops.csv**: all plotting functions of the matplotlib library
- **seaborn_ops.csv**: all plotting functions of the seaborn library

## Workflow

This is the pipeline to download and process all the data:

1. **search_notebooks.py**: search GitHub for potential Jupyter notebooks
2. **download_notebooks.py**: download notebooks from GitHub
3. **ipynb_to_csv.py**: convert all notebooks into 2 dataframes
4. **db_to_csv.py**: convert SQLite database to csv file
5. **analyze_code.py**: analyze code cells, extracting metrics

Additional programs:

- **check_csv.py**: check every repository for csv files
- **parquet_converter.py**: compress the dataframe from csv to parquet
- **recreate_notebooks.py**: recreates all notebooks from the dataset

## Tunable Parameters

To obtain more or different results, these things can be tuned:

- **Pandas operations set** (pandas_ops.csv): currently it contains only 136 operations of Dataframe and general functions. It can be extended, for instance, with index objects, window objects, and so on.
- **Query parameters** (in search_notebooks.py): these parameters are already good, but can be extended.
