
import scanpy as sc
import pickle
import json
import sys


pickle_pth=sys.argv[1]
data_pth=sys.argv[2]

try:
    json_pth=sys.argv[3]
except:
    json_pth=pickle_pth.replace(".pkl", ".json")



with open(pickle_pth, 'rb') as f: 
    causal_graph=pickle.load(f)
real_cells=sc.read_h5ad(data_pth)
gene_idx=real_cells.to_df().columns.tolist()
gene_idx_to_name={gene_idx[key]: [gene_idx[j] for j in values] for key, values in causal_graph.items()}
with open(json_pth, 'w') as f: 
    json.dump(gene_idx_to_name, f,  indent=4)

