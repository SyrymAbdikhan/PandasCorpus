import pandas as pd


def csv_to_parquet(file_path, output_path=None):
    df = pd.read_csv(file_path)
    output_path = (output_path or file_path.replace('.csv', '')) + '.parquet.gzip'
    df.to_parquet(output_path, compression='gzip')


def parquet_to_csv(file_path, output_path=None):
    df = pd.read_parquet(file_path)
    output_path = (output_path or file_path.replace('.parquet.gzip', '')) + '.csv'
    df.to_csv(output_path)


# csv_to_parquet('source/code_cells.csv')
# csv_to_parquet('source/markdown_cells.csv')
# csv_to_parquet('source/csv_files.csv')

parquet_to_csv('source/code_cells.parquet.gzip')
parquet_to_csv('source/markdown_cells.parquet.gzip')
parquet_to_csv('source/csv_files.parquet.gzip')
