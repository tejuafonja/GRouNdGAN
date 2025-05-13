import sys
import scanpy as sc
import pandas as pd

filepath=sys.argv[1]

filepath=filepath.replace('.h5ad','')
adata=sc.read_h5ad(f'{filepath}.h5ad')
dataframe = pd.DataFrame(adata.X, columns=adata.var_names)
dataframe.to_csv(f'{filepath}.csv')