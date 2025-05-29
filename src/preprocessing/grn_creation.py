import pickle
from configparser import ConfigParser
from itertools import chain

import pandas as pd
import scanpy as sc
from arboreto.algo import grnboost2
from tabulate import tabulate
from scipy.sparse import issparse
import os, random

from collections import defaultdict



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
    
    try:
        TFs = pd.read_csv(cfg.get("GRN Preparation", "TFs"), sep="\t")["Symbol"]
        TFs = list(set(TFs).intersection(gene_names))

        #@teju
        if len(TFs) == 0:
            TFs = pd.read_csv(cfg.get("GRN Preparation", "TFs"), sep="\t")["Ensembl"]
            TFs = list(set(TFs).intersection(gene_names))
    except:
        pass

    if issparse(real_cells.X):
        real_cells.X = real_cells.X.toarray()

    # preparing GRNBoost2's input
    real_cells_df = pd.DataFrame(real_cells.X, columns=real_cells.var_names)

    # create directory if not exist.
    os.makedirs(os.path.dirname(cfg.get("GRN Preparation", "Inferred GRN")), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.get("Data", "causal graph")), exist_ok=True)

    seed = int(cfg.get("GRN Preparation", "GRNBoost2 seed"))
    if not os.path.exists(cfg.get("GRN Preparation", "Inferred GRN")):
        # we can optionally pass a list of TFs to GRNBoost2
        print(f"Using {len(TFs)} TFs for GRN inference.")
        real_grn = grnboost2(real_cells_df, tf_names=TFs, verbose=True, seed=seed)
        real_grn.to_csv(cfg.get("GRN Preparation", "Inferred GRN"))

    
    # @teju
    try:
        causal_inference_method = cfg.get("GRN Preparation", "causal_inference_method")
    except:
        causal_inference_method = None # default to 'grnb2'

    if causal_inference_method == 'deep_sem' or causal_inference_method == 'pidc':
        real_grn = create_GRN_extended(cfg)
    else:
        # read GRN csv output, group TFs regulating genes, sort by importance
        real_grn = (
            pd.read_csv(cfg.get("GRN Preparation", "Inferred GRN"))
            .sort_values("importance", ascending=False)
            .astype(str)
        )


    causal_graph = dict(real_grn.groupby("target")["TF"].apply(list))

    k = int(cfg.get("GRN Preparation", "k"))

    # @teju
    try:
        sample_from = cfg.get("GRN Preparation", "sample_from")
    except:
        sample_from = None # default to 'top'

    # @teju
    if sample_from == 'bottom':
        # @teju: Bottom tfs
        causal_graph = {
            gene: set(tfs[-k:])  # to sample the bottom k edges
            for (gene, tfs) in causal_graph.items()
        }
    elif sample_from == 'random':
        # @teju: Random tfs
        random.seed(seed)
        causal_graph_random = {}
        for (gene, tfs) in causal_graph.items():
            tfs_sfl = tfs[:] # to copy without mistakenly shuffling original list.
            random.shuffle(tfs_sfl) # to sample the random k edges
            causal_graph_random[gene] = set(tfs_sfl[:k])
        causal_graph = causal_graph_random
    else:
        causal_graph = {
        gene: set(tfs[:k])  # to sample the top k edges
        # gene: set(tfs[0:10:2]) # sample even indices
        # gene: set(tfs[1:10:2]) # sample odd indices
        for (gene, tfs) in causal_graph.items()
    }
    

    

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


def create_GRN_extended(cfg: ConfigParser) -> None:
    # Find TFs that are in highly variable genes.
    real_cells = sc.read_h5ad(cfg.get("Data", "train"))
    real_grn = pd.read_csv(cfg.get("GRN Preparation", "Inferred GRN"), ',')
    k = int(cfg.get("GRN Preparation", "k"))

    real_grn = real_grn.sort_values(by="importance", ascending=False)
    
    try:
        num_tfs = int(cfg.get("GRN Preparation", "Number of TFs"))
    except:
        num_tfs = None

    try:
        selection_strategy = cfg.get("GRN Preparation", "selection_strategy")
    except:
        selection_strategy = "iterative"

    if num_tfs is None:
        # Building a post-hoc filtered gene regulatory network (GRN) 
        # from a ranked edge list (TF, target, importance).
        #Approaches:
            # 1. naive:
                # Simply picking the top TFs by frequency or rank (e.g., gene1 → gene2 and gene2 → gene1) results in symmetry.
                # So gene2 might be chosen as a TF right after gene1, meaning gene1 can’t use gene2 as a target anymore.
                # That forces us to select low-importance interactions, weakening the inferred network.
            # 2. iterative:
                # Build a mapping of TF → candidate target list (sorted by importance).
                # Select the next TF not already used as a target by first 
                # selecting its top-k target genes.
                # Mark those targets as used.
                # Key rules
                # ========
                # 1. TFs are selected iteratively.
                # 2. A TF cannot be a target of any previously selected TF.
                # 3. A TF cannot have any of its targets also be a TF.

        gene_names = real_cells.var_names.tolist()
        TFs = pd.read_csv(cfg.get("GRN Preparation", "TFs"), sep="\t")["Symbol"]
        TFs = list(set(TFs).intersection(gene_names))
        
    else:
        if selection_strategy == "naive":
            TFs = []
            
            for tf in real_grn.TF.tolist():
                if tf not in TFs and len(TFs) != num_tfs:
                    TFs.append(tf)
            
        else:
            # Group targets by TF
            tf_to_targets = defaultdict(list)
            for _, row in real_grn.iterrows():
                tf_to_targets[row["TF"]].append((row["target"], row["importance"]))

            TFs = []
            used_targets = set()

            for tf in tf_to_targets:
                if len(TFs) >= num_tfs:
                    break

                # Filter out if the next proposed TF is already a target of another TF.
                if tf in used_targets:
                    continue
                
                # Filter out the targets of the next proposed TF that is already a TF.
                targets = [(t, imp) for (t, imp) in tf_to_targets[tf] if t not in TFs]

                if targets:
                    TFs.append(tf)
                    top_targets = targets[:k]
                    used_targets.update([t for t, _ in top_targets])

    
    real_grn = real_grn[real_grn.TF.isin(TFs)]
    return real_grn