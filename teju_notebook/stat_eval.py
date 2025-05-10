import scanpy as sc
import pandas as pd
import numpy as np

import sys
sys.path.append("../")
from teju_utils.stats_metrics import *
from teju_utils.plots import *


import glob


def extract_names(f):
    if "results" in f:
        split=f.split("/results/")[1].split("/")
        
        stage='1' if split[2] == 'stage1' else '2'
        # print(split, stage)
        dict = {
            "tag": "fake",
            "fake_path": f,
            "dataset":split[0],
            "cross_val":split[1].strip("CrossVal_"),
            "knowledge_base":split[2].strip("KB_"),
            "stage":stage,
            "freq":split[3] if stage == '2' else None,
            "method":split[4] if stage == '2' else None,
            "seed":split[5].strip("Seed") if stage == '2' else None,
            "TF":split[6].strip("TF") if stage == '2' else None,
            "simulated":split[8] if stage == '2' else split[4]
        }
    else:
        dict= {}
    return dict



PROJECT_DIR="/p/project1/hai_steerllms/GRouNdGAN/"
df=[]

for dataset in ["PBMC", "PBMC_CTL", "BoneMarrow", "LinearUniform"]: 
# for dataset in ["LinearUniform"]:
    for cross_val in ["CrossVal_1000", "CrossVal_2000"]:
        files=glob.glob(f'{PROJECT_DIR}/results/{dataset}/{cross_val}/**/*.h5ad', recursive=True)
        for f in files:
            # print(f)
            dict=extract_names(f)
            # print(dict)
            if dataset=='PBMC':
                dict['train_path']=f"{PROJECT_DIR}/data/processed/PBMC/{cross_val}/PBMC68k_train.h5ad"
                dict['test_path']=f'{PROJECT_DIR}/data/processed/PBMC/{cross_val}/PBMC68k_test.h5ad'
                dict['valid_path']=f'{PROJECT_DIR}/data/processed/PBMC/{cross_val}/PBMC68k_validation.h5ad'
            elif dataset == 'PBMC_CTL':
                dict['train_path']=f"{PROJECT_DIR}/data/processed/PBMC_CTL/{cross_val}/PBMC_CTL20k_train.h5ad"
                dict['test_path']=f'{PROJECT_DIR}/data/processed/PBMC_CTL/{cross_val}/PBMC_CTL20k_test.h5ad'
                dict['valid_path']=f'{PROJECT_DIR}/data/processed/PBMC_CTL/{cross_val}/PBMC_CTL20k_validation.h5ad'
            elif dataset == 'BoneMarrow':
                dict['train_path']=f"{PROJECT_DIR}/data/processed/BoneMarrow/{cross_val}/BoneMarrow2k_train.h5ad"
                dict['test_path']=f'{PROJECT_DIR}/data/processed/BoneMarrow/{cross_val}/BoneMarrow2k_test.h5ad'
                dict['valid_path']=f'{PROJECT_DIR}/data/processed/BoneMarrow/{cross_val}/BoneMarrow2k_validation.h5ad'
            elif dataset == 'LinearUniform':
                dict['train_path']=f"{PROJECT_DIR}/data/processed/LinearUniform/{cross_val}/LinearUniform5k_train.h5ad"
                dict['test_path']=f'{PROJECT_DIR}/data/processed/LinearUniform/{cross_val}/LinearUniform5k_test.h5ad'
                dict['valid_path']=f'{PROJECT_DIR}/data/processed/LinearUniform/{cross_val}/LinearUniform5k_validation.h5ad'
            
            df.append(dict)

df=pd.DataFrame(df)


df_partial=df.query("stage=='1'")
df_partial=df_partial[~df_partial.duplicated(subset=['cross_val','train_path','test_path'])]


metric='perc_zeros'
df[metric+'_fake'] = df['fake_path'].apply(lambda x: calculate_perc_zeros(x))
df[metric+'_train'] = df_partial['train_path'].apply(lambda x: calculate_perc_zeros(x))
df[metric+'_test'] = df_partial['test_path'].apply(lambda x: calculate_perc_zeros(x))
df[metric+'_valid'] = df_partial['valid_path'].apply(lambda x: calculate_perc_zeros(x))
print('done')

metric='cosine_distance'
# df[metric+'_fake_train'] = df.apply(lambda x: compute_cosine(x['train_path'], x['fake_path']), axis=1)
df[metric+'_fake_test'] = df.apply(lambda x: compute_cosine(x['test_path'], x['fake_path']), axis=1)
df_partial[metric+'_train_test'] = df_partial.apply(lambda x: compute_cosine(x['test_path'], x['train_path']), axis=1)
print('done')


metric='euclidean_distance'
# df[metric+'_fake_train'] = df.apply(lambda x: compute_euclidean(x['train_path'], x['fake_path']), axis=1)
df[metric+'_fake_test'] = df.apply(lambda x: compute_euclidean(x['test_path'], x['fake_path']), axis=1)
df_partial[metric+'_train_test'] = df_partial.apply(lambda x: compute_euclidean(x['test_path'], x['train_path']), axis=1)
print('done')

metric='random_forest'
# df[metric+'_fake_train'] = df.apply(lambda x: compute_auroc_rf(x['train_path'], x['fake_path']), axis=1)
df[metric+'_fake_test'] = df.apply(lambda x: compute_auroc_rf(x['test_path'], x['fake_path']), axis=1)
df_partial[metric+'_train_test'] = df_partial.apply(lambda x: compute_auroc_rf(x['test_path'], x['train_path']), axis=1)
print('done')

metric='mmd'
# df[metric+'_fake_train'] = df.apply(lambda x: compute_mmd(x['train_path'], x['fake_path']), axis=1)
df[metric+'_fake_test'] = df.apply(lambda x: compute_mmd(x['test_path'], x['fake_path']), axis=1)
df_partial[metric+'_train_test'] = df_partial.apply(lambda x: compute_mmd(x['test_path'], x['train_path']), axis=1)
print('done')

df.to_csv("df_results_30.09.csv",index=None)
df_partial.to_csv("df_partial_results_30.09.csv",index=None)