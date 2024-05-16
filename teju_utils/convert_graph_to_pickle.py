import scanpy as sc
import pickle
import json
import sys


json_pth=sys.argv[1]
data_pth=sys.argv[2]

try:
    pickle_pth=sys.argv[3]
except:
    pickle_pth=json_pth.replace(".json", ".pkl")



real_cells=sc.read_h5ad(data_pth)
gene_idx=real_cells.to_df().columns
with open(json_pth, 'r') as f: 
    causal_graph=json.load(f)

causal_graph = {
        gene_idx.get_loc(gene): {gene_idx.get_loc(tf) for tf in tfs}
        for (gene, tfs) in causal_graph.items()
    }
with open(pickle_pth, "wb") as fp:
        pickle.dump(causal_graph, fp, protocol=pickle.HIGHEST_PROTOCOL)

# python convert_graph_to_pickle.py  data/processed/PBMC/causal_graph_TF10_GPT-0.json data/processed/PBMC/PBMC68k_train.h5ad 