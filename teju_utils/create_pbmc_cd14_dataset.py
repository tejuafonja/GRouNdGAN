import scanpy as sc
import os
import pandas as pd


ROOT_DIR='/p/project1/hai_steerllms'
anndata = sc.read_10x_mtx(
           f'{ROOT_DIR}/GRouNdGAN/data/raw/PBMC/' , make_unique=True, gex_only=True
        )

annotations = pd.read_csv(
        f'{ROOT_DIR}/GRouNdGAN/data/raw/PBMC/barcodes_annotations.tsv', delimiter="\t"
    )
annotation_dict = {
    item["barcodes"]: item["celltype"]
    for item in annotations.to_dict("records")
}
anndata.obs["barcodes"] = anndata.obs.index
anndata.obs["celltype"] = anndata.obs["barcodes"].map(annotation_dict)

anndata_ctl = anndata[anndata.obs.celltype == 'CD14+ Monocyte']

os.makedirs(f'{ROOT_DIR}/GRouNdGAN/data/raw/PBMC_CD14', exist_ok=True)
os.makedirs(f'{ROOT_DIR}/GRouNdGAN/data/processed/PBMC_CD14', exist_ok=True)

pth=f'{ROOT_DIR}/GRouNdGAN/data/raw/PBMC_CD14/PBMC2k.h5ad'
anndata_ctl.write_h5ad(pth)