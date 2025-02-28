import pandas as pd
import glob

# 全ファイルを読み込んで結合
files = sorted(glob.glob('cell_gene/cell_gene_dim_1_num_*.csv.zip'))
combined_df = pd.concat([pd.read_csv(f, index_col=0) for f in files], axis=1)

combined_df.to_csv('cell_exprs.csv.gz', compression='gzip')

