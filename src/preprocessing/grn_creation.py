import pickle
from configparser import ConfigParser
from itertools import chain

import pandas as pd
import scanpy as sc
from arboreto.algo import grnboost2
from tabulate import tabulate
from scipy.sparse import issparse
import os


def create_GRN(cfg: ConfigParser) -> None:
    """
    Infers a GRN using GRNBoost2 and uses it to construct a causal graph to impose onto GRouNdGAN.

    Parameters
    ----------
    cfg : ConfigParser
        Parser for config file containing GRN creation params.
    """
    real_cells = sc.read_h5ad(cfg.get("Data", "train"))
    real_cells_val = sc.read_h5ad(cfg.get("Data", "validation"))
    real_cells_test = sc.read_h5ad(cfg.get("Data", "test"))

    # find TFs that are in highly variable genes
    gene_names = real_cells.var_names.tolist()
    TFs = pd.read_csv(cfg.get("GRN Preparation", "TFs"), sep="\t")["Symbol"]
    TFs = list(set(TFs).intersection(gene_names))

    if issparse(real_cells.X):
        real_cells.X = real_cells.X.toarray()

    # preparing GRNBoost2's input
    real_cells_df = pd.DataFrame(real_cells.X, columns=real_cells.var_names)

    # we can optionally pass a list of TFs to GRNBoost2
    print(f"Using {len(TFs)} TFs for GRN inference.")
    # create directory if not exist.
    os.makedirs(os.path.dirname(cfg.get("GRN Preparation", "Inferred GRN")), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.get("Data", "causal graph")), exist_ok=True)

    if not os.path.exists(cfg.get("GRN Preparation", "Inferred GRN")):
        seed = int(cfg.get("GRN Preparation", "GRNBoost2 seed"))
        real_grn = grnboost2(real_cells_df, tf_names=TFs, verbose=True, seed=seed)
        real_grn.to_csv(cfg.get("GRN Preparation", "Inferred GRN"))

    # read GRN csv output, group TFs regulating genes, sort by importance
    real_grn = (
        pd.read_csv(cfg.get("GRN Preparation", "Inferred GRN"))
        .sort_values("importance", ascending=False)
        .astype(str)
    )
    causal_graph = dict(real_grn.groupby("target")["TF"].apply(list))

    k = int(cfg.get("GRN Preparation", "k"))
    causal_graph = {
        gene: set(tfs[:k])  # to sample the top k edges
        # gene: set(tfs[0:10:2]) # sample even indices
        # gene: set(tfs[1:10:2]) # sample odd indices
        for (gene, tfs) in causal_graph.items()
    }
    
    # Temporary to test bottom tfs
    # causal_graph = {
    #     gene: set(tfs[-k:])  # to sample the bottom k edges
    #     # gene: set(tfs[0:10:2]) # sample even indices
    #     # gene: set(tfs[1:10:2]) # sample odd indices
    #     for (gene, tfs) in causal_graph.items()
    # }
    # 

    # get gene, TF names
    regulators = list(chain.from_iterable(causal_graph.values()))
    tfs = set(regulators)

    # delete targets that are also regulators
    causal_graph = {k: v for (k, v) in causal_graph.items() if k not in tfs}

    # get gene, TF names
    regulators = list(chain.from_iterable(causal_graph.values()))
    tfs = set(regulators)

    targets = set(causal_graph.keys())
    genes = sorted(list(tfs | targets))
    # genes = list(tfs | targets)
    # @teju: without sorting, there's no guarantee of reproducing the exact ordering of the list
    # this is useful if you want to map from index back to gene name

    # overwrite train, validation, and test datasets in case there some genes were excluded from the dataset
    real_cells = real_cells[:, genes]

    # real_cells.write_h5ad(cfg.get("Data", "train"))
    # real_cells_val[:, genes].write_h5ad(cfg.get("Data", "validation"))
    # real_cells_test[:, genes].write_h5ad(cfg.get("Data", "test"))

    # print causal graph info
    print(
        "",
        "Causal Graph",
        tabulate(
            [
                ("TFs", len(tfs)),
                ("Targets", len(targets)),
                ("Genes", len(genes)),
                ("Possible Edges", len(tfs) * len(targets)),
                ("Imposed Edges", k * len(targets)),
                ("GRN density Edges", k * len(targets) / (len(tfs) * len(targets))),
            ]
        ),
        sep="\n",
    )

    # save causal graph
    import json

    with open(cfg.get("Data", "causal graph").replace(".pkl", ".json"), "w") as f:
        json.dump(
            {key: sorted(list(values)) for key, values in causal_graph.items()},
            f,
            indent=4,
        )

    # Read the cells again to make sure indices work as intended
    real_cells = sc.read_h5ad(cfg.get("Data", "train"))
    gene_idx = real_cells.to_df().columns

    # convert gene names to numerical indices
    causal_graph = {
        gene_idx.get_loc(gene): {gene_idx.get_loc(tf) for tf in tfs}
        for (gene, tfs) in causal_graph.items()
    }

    # save causal graph
    with open(cfg.get("Data", "causal graph"), "wb") as fp:
        pickle.dump(causal_graph, fp, protocol=pickle.HIGHEST_PROTOCOL)
