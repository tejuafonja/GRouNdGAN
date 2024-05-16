import json
import sys
import scanpy as sc
import pandas as pd

json_pth=sys.argv[1]
data_pth=sys.argv[2]
tf_pth=sys.argv[3]

with open(json_pth, 'r') as f:
    causal_graph=json.load(f)
    


TFS=set([j for values in causal_graph.values() for j in values])
print(f"Total Genes: {len(causal_graph.keys())}")
print(f"Total TFs: {len(TFS)}")

#  assert TFs in Groundtruth TFs
real_cells = sc.read_h5ad(data_pth)
gene_names = real_cells.var_names.tolist()
TFS_gt = pd.read_csv(tf_pth, sep="\t")["Symbol"]
TFS_gt = set(TFS_gt).intersection(gene_names)

# print(TFS_gt - TFS)
assert len(TFS) == len(TFS_gt), 'TF length not equal to Ground-truth.'
assert  len(TFS_gt - TFS) == 0, 'TF not equal to Ground-truth.'

# python check_tf.py data/processed/PBMC/causal_graph_TF10_GPT-0.json data/processed/PBMC/PBMC68k_test.h5ad data/raw/Homo_sapiens_TF.csv 