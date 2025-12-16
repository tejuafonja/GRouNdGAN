import os
import typing

import scanpy as sc
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import issparse
import anndata

class SCDataset(Dataset):
    def __init__(self, path: typing.Union[str, bytes, os.PathLike, anndata._core.anndata.AnnData
]) -> None:
        """
        Create a dataset from the h5ad processed data. Use the
        preprocessing/preprocess.py script to create the h5ad train,
        test, and validation files.

        Parameters
        ----------
        path : typing.Union[str, bytes, os.PathLike]
            Path to the h5ad file.
        """

        print(path)

        if type(path) == str:
            self.data = sc.read_h5ad(path)
        else:
            self.data = path 
            
        if issparse(self.data.X):
            self.data.X = self.data.X.toarray()

        try:
            self.cells = torch.from_numpy(self.data.X)
        except:
            # for some weird reason, the data.X is not overwritten
            self.cells = torch.from_numpy(self.data.X.toarray())
        
        try:    
            # self.cells = torch.from_numpy(self.data.X)
            self.clusters = torch.from_numpy(
                self.data.obs.cluster.to_numpy(dtype=int)
            )
        except:
            # added by teju
            # for dataset that don't use groundgan preprocessing.
            # self.cells = torch.from_numpy(self.data.X.astype('float32'))
            try:
                self.clusters = torch.from_numpy(
                    self.data.obs['RNA_snn_res.0.4'].to_numpy(dtype=int)
                )
            except:
                # ugly I know, but somewhere in the groundgan pipeline, it does expects to return tuple of data,cluster.
                try:
                    self.clusters = torch.from_numpy(
                        self.data.obs['leiden'].to_numpy(dtype=int)
                    )
                except:
                    # cluster assuming self.data has other object lovain needs. This is a workaround and is not ideal. This is why there is a preprocessing step.
                    ann_clustered = self.data.copy()
                    sc.tl.louvain(ann_clustered, resolution=0.15)
                    self.data.obs["cluster"] = ann_clustered.obs["louvain"]
                    self.clusters = torch.from_numpy(
                        self.data.obs.cluster.to_numpy(dtype=int)
                    )

            # 
        # RNA_snn_res.0.4


    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        index : int

        Returns
        -------
        typing.Tuple[torch.Tensor, torch.Tensor]
            Gene expression, Cluster label Tensor tuple.
        """
        return self.cells[index], self.clusters[index]

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            Number of samples (cells).
        """
        return self.cells.shape[0]


def get_loader(
    file_path: typing.Union[str, bytes, os.PathLike],
    batch_size: typing.Optional[int] = None,
) -> DataLoader:
    """
    Provides an IterableLoader over a scRNA-seq Dataset read from given h5ad file.

    Parameters
    ----------
    file_path : typing.Union[str, bytes, os.PathLike]
        Path to the h5ad file.
    batch_size : typing.Optional[int]
        Training batch size. If not specified, the entire dataset
        is returned at each load.

    Returns
    -------
    DataLoader
        Iterable data loader over the dataset.
    """
    dataset = SCDataset(file_path)

    # return the whole dataset if batch size if not specified
    if batch_size is None:
        batch_size = len(dataset)

    return DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
