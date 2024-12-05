#!/usr/bin/python
from __future__ import absolute_import

import os
import shutil

import random
import numpy as np
import torch


import numpy as np
import scanpy as sc

from custom_parser import get_argparser, get_configparser
from factory import get_factory
from preprocessing import grn_creation, preprocess

if __name__ == "__main__":
    """
    Main script to process the data and/or start the training or
    generate cells from an existing model.
    """
    argparser = get_argparser()
    args = argparser.parse_args()

    cfg_parser = get_configparser()
    cfg_parser.read(args.config)
    
    ### Random seed
    random_seed=int(cfg_parser.get("EXPERIMENT", "random_seed"))
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # copy the config file to the output dir
    output_dir = cfg_parser.get("EXPERIMENT", "output directory")
    os.makedirs(output_dir, exist_ok=True)

    try:
        shutil.copy(args.config, output_dir)
    except shutil.SameFileError:
        pass

    # get the GAN factory
    fac = get_factory(cfg_parser)

    if args.preprocess:
        preprocess.preprocess(cfg_parser)

    if args.create_grn:
        grn_creation.create_GRN(cfg_parser)
                    
    if args.train:
        fac.get_trainer()()

    if args.generate:
        n = int(cfg_parser.get("Generation", "number of times to generate"))
        for i in range(n):
            simulated_cells = fac.get_gan().generate_cells(
                int(cfg_parser.get("Generation", "number of cells to generate")),
                checkpoint=cfg_parser.get("EXPERIMENT", "checkpoint"),
            )

            simulated_cells = sc.AnnData(simulated_cells)
            simulated_cells.obs_names = np.repeat("fake", simulated_cells.shape[0])
            simulated_cells.obs_names_make_unique()

            # edited by teju
            real_cells = sc.read_h5ad(cfg_parser.get("Data", "train"))
            gene_names = real_cells.var_names.tolist()
            simulated_cells.var_names=gene_names
            
            
            try:
                assert n == 1
                import os
                os.makedirs(os.path.dirname(cfg_parser.get("EXPERIMENT", "simulated path")), exist_ok=True)
                simulated_cells.write(
                    cfg_parser.get("EXPERIMENT", "simulated path")
                )
            except:
                simulated_cells.write(
                cfg_parser.get("EXPERIMENT", "output directory") + f"/simulated_{i}.h5ad"
            )
    
    if args.generate_cc:
        n = int(cfg_parser.get("Generation", "number of times to generate"))
        for i in range(n):
            simulated_cells = fac.get_cc().generate_cells(
                int(cfg_parser.get("Generation", "number of cells to generate")),
                checkpoint=cfg_parser.get("EXPERIMENT", "checkpoint"),
            )

            simulated_cells = sc.AnnData(simulated_cells)
            simulated_cells.obs_names = np.repeat("fake", simulated_cells.shape[0])
            simulated_cells.obs_names_make_unique()

            # edited by teju
            real_cells = sc.read_h5ad(cfg_parser.get("Data", "train"))
            gene_names = real_cells.var_names.tolist()
            simulated_cells.var_names=gene_names
            
            try:
                assert n == 1
                import os
                os.makedirs(os.path.dirname(cfg_parser.get("EXPERIMENT", "simulated path")), exist_ok=True)
                simulated_cells.write(
                    cfg_parser.get("EXPERIMENT", "simulated path")
                )
            except:
                simulated_cells.write(
                cfg_parser.get("EXPERIMENT", "output directory") + f"_CC/simulated_{i}.h5ad"
            )
    
    if args.grnboost2_exp_runs:
        for new_seed in [1, 2, 3, 4, 5]:
            old_seed = cfg_parser.get("GRN Preparation", "GRNBoost2 seed")
            cfg_parser.set("GRN Preparation", "GRNBoost2 seed", f"{new_seed}")
            
            inferred_grn_pth = cfg_parser.get("GRN Preparation", "Inferred GRN").replace(f"_seed{old_seed}", f"_seed{new_seed}")
            cfg_parser.set("GRN Preparation", "Inferred GRN", inferred_grn_pth)
            
            causal_graph_pth = cfg_parser.get("Data", "causal graph").replace(f"_seed{old_seed}", f"_seed{new_seed}")
            cfg_parser.set("Data", "causal graph", causal_graph_pth)
            
            grn_creation.create_GRN(cfg_parser)
