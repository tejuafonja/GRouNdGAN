#!/usr/bin/python
from __future__ import absolute_import

import os
import typing
import copy

import numpy as np
import torch


import numpy as np
import scanpy as sc

from scipy.sparse import issparse

from torch.utils.data.dataloader import DataLoader

from gans.gan import GAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from torch.nn.utils import parameters_to_vector, vector_to_parameters
from privacy_accountant.RDP_moment_accountant import compute_epsilon, get_noise_multiplier


class DPGAN(GAN):

    def __init__(self,
                genes_no: int,
                batch_size: int,
                latent_dim: int,
                gen_layers: typing.List[int],
                crit_layers: typing.List[int],
                device: typing.Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
                library_size: typing.Optional[int] = 20000,
    ):
    
        super().__init__(
            genes_no=genes_no,
            batch_size=batch_size,
            latent_dim=latent_dim,
            gen_layers=gen_layers,
            crit_layers=crit_layers,
            device=device,
            library_size=library_size,
        )

    def _generate_tsne_plot(
        self,
        valid_loader: DataLoader,
        output_dir: typing.Union[str, bytes, os.PathLike],
    ) -> None:
        """
        Generates t-SNE plots during training.

        Parameters
        ----------
        valid_loader : DataLoader
            Validation set DataLoader.
        output_dir : typing.Union[str, bytes, os.PathLike]
            Directory to save the t-SNE plots.
        """
        
        tsne_path = output_dir + "/TSNE"
        if not os.path.isdir(tsne_path):
            os.makedirs(tsne_path)

        fake_cells = self.generate_cells(len(valid_loader.dataset))
        valid_cells, _ = next(iter(valid_loader))

        embedded_cells = TSNE().fit_transform(
            np.concatenate((valid_cells, fake_cells), axis=0)
        )

        real_embedding = embedded_cells[0 : valid_cells.shape[0], :]
        fake_embedding = embedded_cells[valid_cells.shape[0] :, :]

        plt.clf()
        fig = plt.figure()

        plt.scatter(
            real_embedding[:, 0],
            real_embedding[:, 1],
            c="blue",
            label="real",
            alpha=0.5,
        )

        plt.scatter(
            fake_embedding[:, 0],
            fake_embedding[:, 1],
            c="red",
            label="fake",
            alpha=0.5,
        )

        plt.grid(True)
        plt.legend(
            loc="lower left", numpoints=1, ncol=2, fontsize=8, bbox_to_anchor=(0, 0)
        )

        plt.savefig(tsne_path + "/step_" + str(self.step) + ".jpg")

        with SummaryWriter(f"{output_dir}/TensorBoard/TSNE") as w:
            w.add_figure("t-SNE plot", fig, self.step)

        plt.close()

    def train_group_dp(self,
        train_files: str,
        valid_files: str,
        critic_iter: int,
        max_steps_per_group: int,
        c_lambda: float,
        beta1: float,
        beta2: float,
        gen_alpha_0: float,
        gen_alpha_final: float,
        crit_alpha_0: float,
        crit_alpha_final: float,
        eps: int,
        max_norm: float,
        delta: float,
        groups_per_round: int,
        total_round: int,
        checkpoint: typing.Optional[typing.Union[str, bytes, os.PathLike, None]] = None,
        output_dir: typing.Optional[str] = "output",
        summary_freq: typing.Optional[int] = 5000,
        plt_freq: typing.Optional[int] = 10000,
        save_feq: typing.Optional[int] = 10000,
        nodp=False
    ):

        self.nodp = nodp

        def should_run(freq):
            return freq > 0 and self.step % freq == 0 and self.step > 0
        
        anndata = sc.read_h5ad(train_files)
        patient_ids = anndata.obs['patient_id'].unique()
        groups = np.array(patient_ids)

        sampling_probability = groups_per_round/len(groups) 
        sigma = get_noise_multiplier(target_epsilon=eps, 
                                        target_delta=delta,
                                        sample_rate=sampling_probability, 
                                        steps=total_round)
        print(f"Sigma: {sigma}, Epsilon: {eps}, Groups: {len(groups)}")

        max_steps_per_round = max_steps_per_group

        for t in range(total_round):
            print(f'Round {t}')
            if not self.nodp:
                sample_groups = np.array(groups)[np.random.choice(
                                        a=[False, True],
                                        size=len(groups),
                                        p=[1-sampling_probability, 
                                            sampling_probability]
                                    )]
            else:
                sample_groups = np.array(groups)
                
            # Initial Round Model 
            wt = parameters_to_vector([p.detach().clone() for p in self.gen.parameters()])  

            print(f"This is round: {t}")
            print(wt[:10])

            # Aggregated clipped parameter: difference between the current model after clipping and
            # previous model
            Agg_Delta_W_clip = torch.zeros_like(wt)

            if len(sample_groups) == 0:
                print('NO Group Sampled')
            else:
                for i, group in enumerate(sample_groups):                    
                    
                    print(f"Training... Round {t} group {group}")
                    print(wt[:10])
               
                    # print(max_steps_per_round)
                    anndata_group = anndata[anndata.obs['patient_id'] == group]
                    w0 = torch.tensor(wt)
                    assert torch.equal(wt, w0)
                    
                    # Reset the generator parameter to the round model's parameter
                    vector_to_parameters(w0.detach().clone(), self.gen.parameters()) 

                    self.train(
                        train_files=anndata_group,
                        valid_files=valid_files,
                        critic_iter=critic_iter,
                        max_steps=max_steps_per_round,
                        c_lambda=c_lambda,
                        beta1=beta1,
                        beta2=beta2,
                        gen_alpha_0=gen_alpha_0,
                        gen_alpha_final=gen_alpha_final,
                        crit_alpha_0=crit_alpha_0,
                        crit_alpha_final=crit_alpha_final,
                        checkpoint=None,
                        summary_freq=0,
                        plt_freq=plt_freq,
                        save_feq=0,
                        output_dir=output_dir+f"/round{t+1}",
                        enable_data_parallel=False
                )
                    max_steps_per_round += max_steps_per_group
                    w1 = parameters_to_vector([p.detach().clone() for p in self.gen.parameters()])

                    delta_w = w1 - w0 
                    l2 = delta_w.norm().item()
                    print(f"L2 Norm: {l2}")
                
                    if not self.nodp:
                        # Clip deltaW
                        delta_w_clip = delta_w/max(1, l2/max_norm)
                    else:
                        delta_w_clip = delta_w

                    # Accumulate
                    Agg_Delta_W_clip += delta_w_clip
                
                
            
            if not self.nodp:
                # Sample Noise:
                noise_vec = torch.normal(
                                mean=0.0,
                                std=max_norm * sigma,
                                size=Agg_Delta_W_clip.shape,
                                device=Agg_Delta_W_clip.device,
                                dtype=Agg_Delta_W_clip.dtype
                            )
            else:
                # Create noise vector of zeros
                noise_vec = torch.zeros_like(Agg_Delta_W_clip)
            
            if not self.nodp:
                # Compute average of aggregated (clipping, add noise)
                noisy_average_Agg_Delta_W_clip = 1/groups_per_round * (Agg_Delta_W_clip + noise_vec)
            else:
                # Compute average of aggregated (clipping, add noise)
                noisy_average_Agg_Delta_W_clip = 1/len(sample_groups) * (Agg_Delta_W_clip + noise_vec)
            
            print("Updating weight:")
            # Final Round Model
            # Apply update: make wT = wt + average_delta            
            wT = wt + noisy_average_Agg_Delta_W_clip

            # Replace the weight of model with wT
            vector_to_parameters(wT.detach().clone(), self.gen.parameters())            

            # if should_run(plt_freq):
            loader, valid_loader = self._get_loaders(train_files, valid_files)
            self._generate_tsne_plot(valid_loader, output_dir)

            if should_run(save_feq):
                self._save(output_dir)
                
            print("done training round", t, flush=True)
            
        
        print("DP Training Done.")
        self._save(output_dir)

        loader, valid_loader = self._get_loaders(train_files, valid_files)
        self._generate_tsne_plot(valid_loader, output_dir)




