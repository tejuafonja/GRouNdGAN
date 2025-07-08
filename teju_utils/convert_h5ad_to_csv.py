import sys
import scanpy as sc
import pandas as pd

filepath=sys.argv[1]
format=sys.argv[2]

filepath=filepath.replace('.h5ad','')
adata=sc.read_h5ad(f'{filepath}.h5ad')
dataframe = pd.DataFrame(adata.X, columns=adata.var_names)

if format == 'genexobs':
    dataframe.T.to_csv(f'{filepath}.csv', index=True)
else:
    dataframe.to_csv(f'{filepath}.csv', index=None)
