import os
import typing

import torch 
import numpy as np

import scanpy as sc
from gans.causal_gan import CausalGAN

from torch.nn.utils import parameters_to_vector, vector_to_parameters

from privacy_accountant.RDP_moment_accountant import compute_epsilon, get_noise_multiplier

def safe_parameters_to_vector(parameters):
    return torch.cat([
        p.contiguous().view(-1)
        for p in parameters
    ])


def safe_vector_to_parameters(vec, parameters):
    pointer = 0
    for param in parameters:
        num_param = param.numel()

        # slice correct portion
        param_slice = vec[pointer:pointer + num_param]

        # reshape safely to original shape
        param.data.copy_(
            param_slice.view_as(param).contiguous()
        )

        pointer += num_param


class DPCausalGAN(CausalGAN):
    def __init__(self,
        genes_no: int,
        batch_size: int,
        latent_dim: int,
        noise_per_gene: int,
        depth_per_gene: int,
        width_per_gene: int,
        cc_latent_dim: int,
        cc_layers: typing.List[int],
        cc_pretrained_checkpoint: str,
        crit_layers: typing.List[int],
        causal_graph: typing.Dict[int, typing.Set[int]],
        labeler_layers: typing.List[int],
        device: typing.Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
        library_size: typing.Optional[int] = 20000,
    ):
    
        super().__init__(
            genes_no=genes_no,
            batch_size=batch_size,
            latent_dim=latent_dim,
            noise_per_gene=noise_per_gene,
            depth_per_gene=depth_per_gene,
            width_per_gene=width_per_gene,
            cc_latent_dim=cc_latent_dim,
            cc_layers=cc_layers,
            cc_pretrained_checkpoint=cc_pretrained_checkpoint,
            crit_layers=crit_layers,
            causal_graph=causal_graph,
            labeler_layers=labeler_layers,
            device=device,
            library_size=library_size
        )

        self.gen = None
        self.crit = None
        self._build_model()

        self.step = 0
        self.gen_opt = None
        self.crit_opt = None
        self.gen_lr_scheduler = None
        self.crit_lr_scheduler = None
    
    def _train_labelers(self, real_cells: torch.Tensor) -> None:
        """
        Trains the labeler (on real and fake) and anti-labeler (on fake only).

        Parameters
        ----------
        real_cells : torch.Tensor
            Tensor containing a batch of real cells.
        """
        fake_noise = self._generate_noise(self.batch_size, self.latent_dim, self.device)
        fake = self.gen(fake_noise).detach() 

        # train anti-labeler
        self.antilabeler_opt.zero_grad()
        try:
            predicted_tfs = self.antilabeler(fake[:, self.gen.module.genes])
        except:
            predicted_tfs = self.antilabeler(fake[:, self.gen.genes])

        # actual_tfs = fake[:, self.gen.module.tfs]
        actual_tfs = fake[:, self.gen.tfs]

        antilabeler_loss = self.mse(predicted_tfs, actual_tfs)
        antilabeler_loss.backward(retain_graph=True)
        self.antilabeler_opt.step()

        # train labeler on fake data
        self.labeler_opt.zero_grad()
        try:
            predicted_tfs = self.labeler(fake[:, self.gen.module.genes])
        except:
            predicted_tfs = self.labeler(fake[:, self.gen.genes])

        labeler_floss = self.mse(predicted_tfs, actual_tfs)
        labeler_floss.backward()
        self.labeler_opt.step()

        # train labeler on real data
        self.labeler_opt.zero_grad()
        try:
            predicted_tfs = self.labeler(real_cells[:, self.gen.module.genes])
        except:
            predicted_tfs = self.labeler(real_cells[:, self.gen.genes])

        # actual_tfs = real_cells[:, self.gen.module.tfs]
        actual_tfs = real_cells[:, self.gen.tfs]
        labeler_rloss = self.mse(predicted_tfs, actual_tfs)
        labeler_rloss.backward()
        self.labeler_opt.step()

    def _train_generator(self) -> torch.Tensor:
        """
        Trains the causal generator for one iteration.
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
        # print(fake.shape, fake_noise.shape)
        # print(len(self.gen.genes))
        # print(self.gen.num_genes, self.gen.num_tfs)


        # noise = torch.randn(self.batch_size, 128, device=self.device)
        # print("NOISE SHAPE", noise.shape )

        # tf_expressions = self.causal_controller(noise)
        # print("--")
        # print(tf_expressions.shape)

        # predicted_tfs = self.labeler(fake[:, self.gen.module.genes])
        
        predicted_tfs = self.labeler(fake[:, self.gen.genes])

        # actual_tfs = fake[:, self.gen.module.tfs]
        actual_tfs = fake[:, self.gen.tfs]
        labeler_loss = self.mse(predicted_tfs, actual_tfs)

        # try:
        # predicted_tfs = self.antilabeler(fake[:, self.gen.module.genes])
        # except:
        predicted_tfs = self.antilabeler(fake[:, self.gen.genes])

        antilabeler_loss = self.mse(predicted_tfs, actual_tfs)

        crit_fake_pred = self.crit(fake)
        gen_loss = self._generator_loss(crit_fake_pred)

        # comment for ablation of labeler and anti-labeler (GRouNdGAN_def_even_ablation1)
        gen_loss += labeler_loss + antilabeler_loss
        
        # uncomment for ablation of anti-labeler but keeping the labeler (GRouNdGAN_def_even_ablation2)
        # gen_loss += labeler_loss

        # uncomment for ablation of labeler but keeping the anti-labeler (GRouNdGAN_def_even_ablation3)
        # gen_loss += antilabeler_loss

        
        gen_loss.backward()

        # Update weights
        self.gen_opt.step()

        return gen_loss


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
                "labeler_state_dict": self.labeler.module.state_dict(),
                "antilabeler_state_dict": self.antilabeler.module.state_dict(),
                "generator_optimizer_state_dict": self.gen_opt.state_dict(),
                "critic_optimizer_state_dict": self.crit_opt.state_dict(),
                "labeler_optimizer_state_dict": self.labeler_opt.state_dict(),
                "antilabeler_optimizer_state_dict": self.antilabeler_opt.state_dict(),
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
                "labeler_state_dict": self.labeler.state_dict(),
                "antilabeler_state_dict": self.antilabeler.state_dict(),
                "generator_optimizer_state_dict": self.gen_opt.state_dict(),
                "critic_optimizer_state_dict": self.crit_opt.state_dict(),
                "labeler_optimizer_state_dict": self.labeler_opt.state_dict(),
                "antilabeler_optimizer_state_dict": self.antilabeler_opt.state_dict(),
                "generator_lr_scheduler": self.gen_lr_scheduler.state_dict(),
                "critic_lr_scheduler": self.crit_lr_scheduler.state_dict(),
            },
            f"{path}/checkpoints/step_{self.step}.pth",
        )


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
        labeler_alpha: float,
        antilabeler_alpha: float,
        labeler_training_interval: int,
        eps: int,
        delta: float,
        max_norm: float,
        groups_per_round: int,
        total_round: int,
        checkpoint: typing.Optional[typing.Union[str, bytes, os.PathLike, None]] = None,
        output_dir: typing.Optional[str] = "output",
        summary_freq: typing.Optional[int] = 1000,
        plt_freq: typing.Optional[int] = 1000,
        save_feq: typing.Optional[int] = 1000,
        max_steps_per_group: typing.Optional[int] = None,
        crit_dp_mode: typing.Optional[str] = 'none',
        nodp: str = "False"
    ):

        # if torch.cuda.device_count() > 1 and self.device.startswith("cuda"):
        #     self.gen = torch.nn.DataParallel(self.gen)
        #     self.crit = torch.nn.DataParallel(self.crit)

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

        def should_run(freq):
            return freq > 0 and self.step % freq == 0 and self.step > 0
        
        def should_run_plot_and_save(freq, round):
            return freq > 0 and round % freq == 0 and self.step > 0


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
            # wt_gen = parameters_to_vector(self.gen.parameters()).detach().clone()
            # wt_crit = parameters_to_vector(self.crit.parameters()).detach().clone()
            # wt_labeler = parameters_to_vector(self.labeler.parameters()).detach().clone()
            # wt_antilabeler = parameters_to_vector(self.antilabeler.parameters()).detach().clone()

            wt_gen = safe_parameters_to_vector(self.gen.parameters()).detach().clone()
            wt_crit = safe_parameters_to_vector(self.crit.parameters()).detach().clone()
            wt_labeler = safe_parameters_to_vector(self.labeler.parameters()).detach().clone()
            wt_antilabeler = safe_parameters_to_vector(self.antilabeler.parameters()).detach().clone()

            print(f"Round model (before training): {wt_gen}")

            Agg_clipped_delta_w_gen = torch.zeros_like(wt_gen)
            Agg_clipped_delta_w_crit = torch.zeros_like(wt_crit)
            Agg_clipped_delta_w_labeler = torch.zeros_like(wt_labeler)
            Agg_clipped_delta_w_antilabeler = torch.zeros_like(wt_antilabeler)
            

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
                    vector_to_parameters(wt_labeler.detach().clone(), self.labeler.parameters())
                    vector_to_parameters(wt_antilabeler.detach().clone(), self.antilabeler.parameters())

                    

                    # Instantiate optimizers
                    self.gen_opt = torch.optim.AdamW(
                        filter(lambda p: p.requires_grad, self.gen.parameters()),
                        lr=gen_alpha_0,
                        betas=(beta1, beta2),
                        amsgrad=True,
                    )

                    self.crit_opt = torch.optim.AdamW(
                        self.crit.parameters(),
                        lr=crit_alpha_0,
                        betas=(beta1, beta2),
                        amsgrad=True,
                    )

                    self.labeler_opt = torch.optim.AdamW(
                        self.labeler.parameters(),
                        lr=labeler_alpha,
                        betas=(beta1, beta2),
                        amsgrad=True,
                    )

                    self.antilabeler_opt = torch.optim.AdamW(
                        self.antilabeler.parameters(),
                        lr=antilabeler_alpha,
                        betas=(beta1, beta2),
                        amsgrad=True,
                    )

                    # for the labeler and anti-labeler
                    self.mse = torch.nn.MSELoss()

                    # Exponential Learning Rate
                    self.gen_lr_scheduler = self._set_exponential_lr(
                        self.gen_opt, gen_alpha_0, gen_alpha_final, max_steps_per_group
                    )
                    self.crit_lr_scheduler = self._set_exponential_lr(
                        self.crit_opt, crit_alpha_0, crit_alpha_final, max_steps_per_group
                    )

                    self.gen.train()
                    self.crit.train()
                    self.labeler.train()
                    self.antilabeler.train()

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

                        if should_run(labeler_training_interval):
                            self._train_labelers(real_cells)

                        # print("done training local step", self.step, flush=True)
                        self.step += 1

                    # Compute client's delta relative to the round-global snapshot
                    # local_params_gen = parameters_to_vector(self.gen.parameters()).detach()
                    # local_params_crit = parameters_to_vector(self.crit.parameters()).detach()
                    # local_params_labeler = parameters_to_vector(self.labeler.parameters()).detach()
                    # local_params_antilabeler = parameters_to_vector(self.antilabeler.parameters()).detach()

                    local_params_gen = safe_parameters_to_vector(self.gen.parameters()).detach()
                    local_params_crit = safe_parameters_to_vector(self.crit.parameters()).detach()
                    local_params_labeler = safe_parameters_to_vector(self.labeler.parameters()).detach()
                    local_params_antilabeler = safe_parameters_to_vector(self.antilabeler.parameters()).detach()
                    
                    delta_w_gen = (local_params_gen - wt_gen).detach()
                    delta_w_crit = (local_params_crit - wt_crit).detach()
                    delta_w_labeler  = (local_params_labeler - wt_labeler).detach()
                    delta_w_antilabeler = (local_params_antilabeler - wt_antilabeler).detach()

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
                    
                    delta_w_clip_labeler = delta_w_labeler
                    delta_w_clip_antilabeler = delta_w_antilabeler


                    # Accumulate (sum of clipped deltas)
                    Agg_clipped_delta_w_gen += delta_w_clip_gen
                    Agg_clipped_delta_w_crit += delta_w_clip_crit
                    Agg_clipped_delta_w_labeler += delta_w_clip_labeler
                    Agg_clipped_delta_w_antilabeler += delta_w_clip_antilabeler
                    
                # -------------------------------
                # Global Model Update
                # -------------------------------
                average_delta_w_labeler = Agg_clipped_delta_w_labeler / float(groups_per_round)
                wT_labeler = wt_labeler + average_delta_w_labeler
                # vector_to_parameters(wT_labeler.detach().clone(), self.labeler.parameters())
                safe_vector_to_parameters(wT_labeler.detach().clone(), self.labeler.parameters())

                average_delta_w_antilabeler = Agg_clipped_delta_w_antilabeler / float(groups_per_round)
                wT_antilabeler = wt_antilabeler + average_delta_w_antilabeler
                # vector_to_parameters(wT_antilabeler.detach().clone(), self.antilabeler.parameters())
                safe_vector_to_parameters(wT_antilabeler.detach().clone(), self.antilabeler.parameters())

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
                    # vector_to_parameters(wT_gen.detach().clone(), self.gen.parameters())
                    safe_vector_to_parameters(wT_gen.detach().clone(), self.gen.parameters())

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
                        # vector_to_parameters(wT_crit.detach().clone(), self.crit.parameters())
                        safe_vector_to_parameters(wT_crit.detach().clone(), self.crit.parameters())
                    else:
                        # If critic DP is not enabled, we do not add noise but still average (if desired)
                        average_delta_w_crit = Agg_clipped_delta_w_crit / float(groups_per_round)
                        wT_crit = wt_crit + average_delta_w_crit
                        # vector_to_parameters(wT_crit.detach().clone(), self.crit.parameters())
                        safe_vector_to_parameters(wT_crit.detach().clone(), self.crit.parameters())
                else:
                    # Non-DP: average by actual sampled count (which equals len(groups) here) and apply
                    average_delta_w_gen = Agg_clipped_delta_w_gen / float(len(sample_groups))
                    wT_gen = wt_gen + average_delta_w_gen
                    # vector_to_parameters(wT_gen.detach().clone(), self.gen.parameters())
                    safe_vector_to_parameters(wT_gen.detach().clone(), self.gen.parameters())

                    average_delta_w_crit = Agg_clipped_delta_w_crit / float(len(sample_groups))
                    wT_crit = wt_crit + average_delta_w_crit
                    # vector_to_parameters(wT_crit.detach().clone(), self.crit.parameters())
                    safe_vector_to_parameters(wT_crit.detach().clone(), self.crit.parameters())
                
                print(f"--- Done Round {t} ---")
                
                # loader, valid_loader = self._get_loaders(train_files, valid_files)
                # self._generate_tsne_plot(valid_loader, output_dir+f"/round{t}")
                # self._save(output_dir)
                if should_run_plot_and_save(plt_freq, t):
                    loader, valid_loader = self._get_loaders(train_files, valid_files)
                    self._generate_tsne_plot(valid_loader, output_dir+f"/round{t}")

                if should_run_plot_and_save(save_feq, t):
                    self._save(output_dir)
        
        loader, valid_loader = self._get_loaders(train_files, valid_files)
        self._generate_tsne_plot(valid_loader, output_dir)
        self._save(output_dir)
        print("DP Training Done.")