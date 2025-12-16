import os
import typing

import torch 
import numpy as np

import scanpy as sc
from gans.gan import GAN

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

        self.gen = None
        self.crit = None
        self._build_model()

        self.step = 0
        self.gen_opt = None
        self.crit_opt = None
        self.gen_lr_scheduler = None
        self.crit_lr_scheduler = None

    def _save(self, path: typing.Union[str, bytes, os.PathLike]) -> None:
        """
        Saves the model.

        Parameters
        ----------
        path : typing.Union[str, bytes, os.PathLike]
            Directory to save the model.
        """
        output_dir = path + "/checkpoints"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        # import pdb; pdb.set_trace()
        try:

            torch.save(
                {
                    "step": self.step,
                    "generator_state_dict": self.gen.module.state_dict(),
                    "critic_state_dict": self.crit.module.state_dict(),
                    "generator_optimizer_state_dict": self.gen_opt.state_dict(),
                    "critic_optimizer_state_dict": self.crit_opt.state_dict(),
                    "generator_lr_scheduler": self.gen_lr_scheduler.state_dict(),
                    "critic_lr_scheduler": self.crit_lr_scheduler.state_dict(),
                },
                f"{path}/checkpoints/step_{self.step}.pth",
            )
        except:
            torch.save(
                {
                    "step": self.step,
                    "generator_state_dict": self.gen.state_dict(),
                    "critic_state_dict": self.crit.state_dict(),
                    "generator_optimizer_state_dict": self.gen_opt.state_dict(),
                    "critic_optimizer_state_dict": self.crit_opt.state_dict(),
                    "generator_lr_scheduler": self.gen_lr_scheduler.state_dict(),
                    "critic_lr_scheduler": self.crit_lr_scheduler.state_dict(),
                },
                f"{path}/checkpoints/step_{self.step}.pth",
            )
    def _get_gradient(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        epsilon: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the gradient of the critic's scores with respect to interpolations
        of real and fake cells.

        Parameters
        ----------
        real : torch.Tensor
            A batch of real cells.
        fake : torch.Tensor
            A batch of fake cells.
        epsilon : torch.Tensor
            A vector of the uniformly random proportions of real/fake per interpolated cells.

        Returns
        -------
        torch.Tensor
            Gradient of the critic's score with respect to interpolated data.
        """

        # Mix real and fake cells together
        interpolates = real * epsilon + fake * (1 - epsilon)

        # Calculate the critic's scores on the mixed data
        critic_interpolates = self.crit(interpolates)

        # Take the gradient of the scores with respect to the data
        gradient = torch.autograd.grad(
            inputs=interpolates,
            outputs=critic_interpolates,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]
        return gradient

    @staticmethod
    def _gradient_penalty(gradient: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient penalty given a gradient.

        Parameters
        ----------
        gradient : torch.Tensor
            The gradient of the critic's score with respect to
            the interpolated data.

        Returns
        -------
        torch.Tensor
            Gradient penalty of the given gradient.
        """
        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)

        return torch.mean((gradient_norm - 1) ** 2)

    def _train_critic(
        self, real_cells: torch.Tensor, real_labels: torch.Tensor, c_lambda: float
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Trains the critic for one iteration.

        Parameters
        ----------
        real_cells : torch.Tensor
            Tensor containing a batch of real cells.
        real_labels : torch.Tensor
            Tensor containing a batch of real labels (corresponding to real_cells).
        c_lambda : float
            Regularization hyper-parameter for gradient penalty.

        Returns
        -------
        typing.Tuple[torch.Tensor, torch.Tensor]
            The computed critic loss and gradient penalty.
        """
        self.crit_opt.zero_grad()

        fake_noise = self._generate_noise(self.batch_size, self.latent_dim, self.device)
        fake = self.gen(fake_noise)

        crit_fake_pred = self.crit(fake.detach())
        crit_real_pred = self.crit(real_cells)

        epsilon = torch.rand(len(real_cells), 1, device=self.device, requires_grad=True)

        gradient = self._get_gradient(real_cells, fake.detach(), epsilon)
        gp = self._gradient_penalty(gradient)

        crit_loss = self._critic_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

        # Update gradients
        crit_loss.backward(retain_graph=True)

        # Update optimizer
        self.crit_opt.step()

        return crit_loss, gp
    
    def _train_generator(self) -> torch.Tensor:
        """
        Trains the generator for one iteration.

        Returns
        -------
        torch.Tensor
            Tensor containing only 1 item, the generator loss.
        """
        self.gen_opt.zero_grad()

        fake_noise = self._generate_noise(
            self.batch_size, self.latent_dim, device=self.device
        )

        fake = self.gen(fake_noise)
        crit_fake_pred = self.crit(fake)

        gen_loss = self._generator_loss(crit_fake_pred)
        gen_loss.backward()

        # Update weights
        self.gen_opt.step()

        return gen_loss

    def train_group_dp(self, 
        train_files: str,
        valid_files: str,
        critic_iter: int,
        c_lambda: float,
        beta1: float,
        beta2: float,
        gen_alpha_0: float,
        gen_alpha_final: float,
        crit_alpha_0: float,
        crit_alpha_final: float,
        eps: int,
        delta: float,
        max_norm: float,
        groups_per_round: int,
        total_round: int,
        max_steps_per_group: typing.Optional[int] = None,
        crit_dp_mode: typing.Optional[str] = 'none',
        checkpoint: typing.Optional[typing.Union[str, bytes, os.PathLike, None]] = None,
        output_dir: typing.Optional[str] = "output",
        summary_freq: typing.Optional[int] = 1000,
        plt_freq: typing.Optional[int] = 1000,
        save_feq: typing.Optional[int] = 1000,
        nodp: str = "False"
    ):

        if torch.cuda.device_count() > 1 and self.device.startswith("cuda"):
            self.gen = torch.nn.DataParallel(self.gen)
            self.crit = torch.nn.DataParallel(self.crit)

        assert nodp in ("False", "True")
        self.nodp = False if nodp == 'False' else True
        
        if crit_dp_mode not in ['dp', 'none']:
            raise ValueError("crit_dp_mode must be 'none', or 'dp'.")
                
        anndata = sc.read_h5ad(train_files)
        patient_ids = anndata.obs['patient_id'].unique()
        groups = np.array(patient_ids)

        sampling_probability = groups_per_round / len(groups)
        sigma = get_noise_multiplier(
            target_epsilon=eps,
            target_delta=delta,
            sample_rate=sampling_probability,
            steps=total_round 
        )
        if self.nodp:
            print(f"Non-DP mode. All {groups_per_round} groups trained per round.")
        else:
            print(f"Sigma: {sigma}, Epsilon: {eps}, Groups:{len(groups)}, Groups per round: {groups_per_round}")

        self.device = "cuda"

        if checkpoint is not None:
            self._load(checkpoint, mode="training")
        
        if self.nodp:
            crit_dp_mode = "none" # Enforce non-DP critic mode

        def should_run(freq, round):
            return freq > 0 and round % freq == 0 and self.step > 0

        self.gen_opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.gen.parameters()),
            lr=gen_alpha_0,  
            betas=(beta1, beta2),
            amsgrad=True
        )

        self.crit_opt = torch.optim.AdamW(
            self.crit.parameters(),
            lr=crit_alpha_0,
            betas=(beta1, beta2),
            amsgrad=True,
        )

        for t in range(total_round):
            print(f"--- Starting Round {t} ---")
            if self.nodp:
                sample_indices = np.random.choice(len(groups), size=groups_per_round, replace=False)
                sample_groups = np.array(groups)[sample_indices]
            else:
                # DP sampling logic
                sample_groups = np.array(groups)[np.random.choice(
                    a=[False, True],
                    size=len(groups),
                    p=[1-sampling_probability, sampling_probability]
                )]

            # Save global model snapshots at start of round
            wt_gen = parameters_to_vector(self.gen.parameters()).detach().clone()
            wt_crit = parameters_to_vector(self.crit.parameters()).detach().clone()

            print(f"Round model (before training): {wt_gen}")

            Agg_clipped_delta_w_gen = torch.zeros_like(wt_gen)
            Agg_clipped_delta_w_crit = torch.zeros_like(wt_crit)
            

            if len(sample_groups) == 0:
                print("No group sampled. Noise will still be added for privacy.")
            else:
                for i, g in enumerate(sample_groups):
                    print(f"--- Sampled groups for round {t} = {len(sample_groups)}---")

                    anndata_group = anndata[anndata.obs['patient_id'] == g]
                    loader, valid_loader = self._get_loaders(anndata_group, valid_files)
                    loader_gen = iter(loader)
                    
                    if max_steps_per_group is None:
                        local_max_steps = len(anndata_group) // self.batch_size
                    else:
                        local_max_steps = max_steps_per_group

                    print(f"Training, round {t}, group:{g}, step_per_group={local_max_steps}")

                    # Reset generator/critic parameter to the round model's parameter
                    vector_to_parameters(wt_gen.detach().clone(), self.gen.parameters())
                    vector_to_parameters(wt_crit.detach().clone(), self.crit.parameters())

                    self.gen.train()
                    self.crit.train()

                    self.gen_opt = torch.optim.AdamW(
                        filter(lambda p: p.requires_grad, self.gen.parameters()),
                        lr=gen_alpha_0,
                        betas=(beta1, beta2),
                        amsgrad=True
                    )

                    self.crit_opt = torch.optim.AdamW(
                        self.crit.parameters(),
                        lr=crit_alpha_0,
                        betas=(beta1, beta2),
                        amsgrad=True,
                    )

                    # Exponential Learning Rate
                    self.gen_lr_scheduler = self._set_exponential_lr(
                        self.gen_opt, gen_alpha_0, gen_alpha_final, max_steps_per_group
                    )
                    self.crit_lr_scheduler = self._set_exponential_lr(
                        self.crit_opt, crit_alpha_0, crit_alpha_final, max_steps_per_group
                    )

                    generator_losses, critic_losses = [], []

                    for local_step in range(local_max_steps):
                        # 1. Get data
                        try:
                            real_cells, _ = next(loader_gen)
                        except StopIteration:
                            loader_gen = iter(loader)
                            real_cells, _ = next(loader_gen)
                        
                        real_cells = real_cells.to(self.device)

                        # 2. Train Critic
                        # Only train critic after step 0 to allow generator to learn first
                        if self.step != 0: 
                            mean_iter_crit_loss = 0
                            for _ in range(critic_iter):
                                crit_loss, gp = self._train_critic(real_cells, None, c_lambda)
                                mean_iter_crit_loss += crit_loss.item() / critic_iter                    
                            critic_losses += [mean_iter_crit_loss]

                            # Update learning rate
                            self.crit_lr_scheduler.step()

                        # 3. Train Generator
                        gen_loss = self._train_generator()
                        self.gen_lr_scheduler.step()

                        generator_losses += [gen_loss.item()]

                        # print("done training local step", self.step, flush=True)
                        self.step += 1

                    # Compute client's delta relative to the round-global snapshot
                    local_params_gen = parameters_to_vector(self.gen.parameters()).detach()
                    local_params_crit = parameters_to_vector(self.crit.parameters()).detach()
                    delta_w_gen = (local_params_gen - wt_gen).detach()
                    delta_w_crit = (local_params_crit - wt_crit).detach()

                    # Clip generator delta
                    if not self.nodp:
                        l2_gen = delta_w_gen.norm(2).item()
                        delta_w_clip_gen = delta_w_gen / max(1.0, l2_gen / max_norm)
                    else:
                        # non-DP: don't clip
                        delta_w_clip_gen = delta_w_gen

                    # Clip critic delta if critic DP enabled
                    if crit_dp_mode == "dp" and not self.nodp:
                        l2_crit = delta_w_crit.norm(2).item()
                        delta_w_clip_crit = delta_w_crit / max(1.0, l2_crit / max_norm)
                    else:
                        delta_w_clip_crit = delta_w_crit

                    # Accumulate (sum of clipped deltas)
                    Agg_clipped_delta_w_gen += delta_w_clip_gen
                    Agg_clipped_delta_w_crit += delta_w_clip_crit
                    
                # -------------------------------
                # Global Model Update
                # -------------------------------
                if not self.nodp:
                    print("ADDING NOISE.")
                    # DP: add Gaussian noise to the sum of clipped deltas, then average by groups_per_round
                    noise_vec_gen = torch.normal(
                                        mean=0.0,
                                        std=max_norm * sigma,
                                        size=Agg_clipped_delta_w_gen.shape,
                                        device=Agg_clipped_delta_w_gen.device,
                                        dtype=Agg_clipped_delta_w_gen.dtype
                                    )         
                    # Compute average of aggregated (clipping, add noise)
                    # # NOTE: Uses groups_per_round as denominator, which is an assumption for DP-FedAvg style.
   
                    noisy_average_delta_w_gen = (Agg_clipped_delta_w_gen + noise_vec_gen) / float(groups_per_round)
                    wT_gen = wt_gen + noisy_average_delta_w_gen
                    vector_to_parameters(wT_gen.detach().clone(), self.gen.parameters())

                    if crit_dp_mode == 'dp':
                        noise_vec_crit = torch.normal(
                            mean=0.0,
                            std=max_norm * sigma,
                            size=Agg_clipped_delta_w_crit.shape,
                            device=Agg_clipped_delta_w_crit.device,
                            dtype=Agg_clipped_delta_w_crit.dtype
                        )
                        noisy_average_delta_w_crit = (Agg_clipped_delta_w_crit + noise_vec_crit) / float(groups_per_round)
                        wT_crit = wt_crit + noisy_average_delta_w_crit
                        vector_to_parameters(wT_crit.detach().clone(), self.crit.parameters())
                    else:
                        # If critic DP is not enabled, we do not add noise but still average (if desired)
                        average_delta_w_crit = Agg_clipped_delta_w_crit / float(groups_per_round)
                        wT_crit = wt_crit + average_delta_w_crit
                        vector_to_parameters(wT_crit.detach().clone(), self.crit.parameters())
                else:
                    # Non-DP: average by actual sampled count (which equals len(groups) here) and apply
                    average_delta_w_gen = Agg_clipped_delta_w_gen / float(len(sample_groups))
                    wT_gen = wt_gen + average_delta_w_gen
                    vector_to_parameters(wT_gen.detach().clone(), self.gen.parameters())

                    average_delta_w_crit = Agg_clipped_delta_w_crit / float(len(sample_groups))
                    wT_crit = wt_crit + average_delta_w_crit
                    vector_to_parameters(wT_crit.detach().clone(), self.crit.parameters())
                
                print(f"--- Done Round {t} ---")
                
                # loader, valid_loader = self._get_loaders(train_files, valid_files)
                # self._generate_tsne_plot(valid_loader, output_dir+f"/round{t}")
                # self._save(output_dir)
                if should_run(plt_freq, t):
                    loader, valid_loader = self._get_loaders(train_files, valid_files)
                    self._generate_tsne_plot(valid_loader, output_dir+f"/round{t}")

                if should_run(save_feq, t):
                    self._save(output_dir)
        
        loader, valid_loader = self._get_loaders(train_files, valid_files)
        self._generate_tsne_plot(valid_loader, output_dir)
        self._save(output_dir)
        print("DP Training Done.")