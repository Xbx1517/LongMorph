"""PyTorch training adapters for optional upstream model implementations.

Source locations are provided by the caller; importing this module does not
import any upstream implementation.
"""

import sys
import types

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from longmorph.baselines.common import seed_torch


def loader(x, batch_size, seed, channels_first=False):
    t = torch.from_numpy(x).float()
    if channels_first:
        t = t.transpose(1, 2)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(t),
        batch_size=min(batch_size, len(t)),
        shuffle=True,
        drop_last=False,
        generator=gen,
    )


def train_tts_gan(train, args, device, out):
    sys.path.insert(0, str(args.vendor_dir / "tts_gan"))
    from GANModels import Generator, Discriminator

    # The public Generator hard-codes five attention heads inside its encoder block.
    G = Generator(
        seq_len=24,
        channels=32,
        latent_dim=100,
        embed_dim=40,
        depth=3,
        num_heads=5,
        forward_drop_rate=0.2,
        attn_drop_rate=0.2,
    ).to(device)
    D = Discriminator(
        in_channels=32,
        patch_size=4,
        emb_size=64,
        seq_length=24,
        depth=3,
        n_classes=1,
        num_heads=4,
    ).to(device)
    go = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    do = torch.optim.Adam(D.parameters(), lr=4e-4, betas=(0.0, 0.9))
    dl = loader(train, args.batch_size, args.seed, channels_first=True)
    history = []
    for epoch in range(args.epochs):
        for (real,) in dl:
            real = real.to(device).unsqueeze(2)
            z = torch.randn(len(real), 100, device=device)
            fake = G(z)
            do.zero_grad(set_to_none=True)
            d_loss = torch.relu(1.0 - D(real)).mean() + torch.relu(1.0 + D(fake.detach())).mean()
            d_loss.backward()
            do.step()
            go.zero_grad(set_to_none=True)
            g_loss = -D(G(torch.randn(len(real), 100, device=device))).mean()
            g_loss.backward()
            go.step()
        history.append([float(d_loss.detach()), float(g_loss.detach())])
    G.eval()
    with torch.no_grad():
        z = torch.randn(args.num_samples, 100, device=device)
        generated = G(z).squeeze(2).transpose(1, 2).cpu().numpy()
    torch.save(
        {
            "generator": G.state_dict(),
            "discriminator": D.state_dict(),
            "history": history,
        },
        out / "models/model.pt",
    )
    return generated, {
        "epochs": args.epochs,
        "architecture": "official TTS-GAN transformer GAN",
    }


def train_diffusion_ts(train, args, device, out):
    sys.path.insert(0, str(args.vendor_dir / "diffusion_ts"))
    from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS

    model = Diffusion_TS(
        seq_length=24,
        feature_size=32,
        n_layer_enc=2,
        n_layer_dec=3,
        d_model=64,
        timesteps=100,
        sampling_timesteps=20,
        loss_type="l1",
        n_heads=4,
        use_ff=True,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    dl = loader(train, args.batch_size, args.seed)
    it = iter(dl)
    history = []
    model.train()
    for step in range(args.steps):
        try:
            (x,) = next(it)
        except StopIteration:
            it = iter(dl)
            (x,) = next(it)
        x = x.to(device)
        opt.zero_grad(set_to_none=True)
        loss = model(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step + 1 == args.steps:
            history.append([step, float(loss.detach())])
    model.eval()
    chunks = []
    with torch.no_grad():
        for n in range(0, args.num_samples, args.batch_size):
            chunks.append(
                model.generate_mts(min(args.batch_size, args.num_samples - n)).cpu().numpy()
            )
    torch.save({"model": model.state_dict(), "history": history}, out / "models/model.pt")
    return np.concatenate(chunks), {
        "steps": args.steps,
        "architecture": "official Diffusion-TS",
    }


def train_fourier_flow(train, args, device, out):
    # The upstream CPU implementation fits one univariate flow per PCA coordinate.
    sys.path.insert(0, str(args.vendor_dir / "fourier_flow"))
    from SequentialFlows import FourierFlow

    # An upstream transitive import calls torch.manual_seed(1); restore the
    # requested seed after the upstream module import.
    seed_torch(args.seed)
    generated = np.empty((args.num_samples, 24, 32), dtype=np.float32)
    states = []
    for j in range(32):
        # The official experiment uses T+1 for even-length series (DFT packs real/imaginary pairs).
        # `normalize=False` is an official option and avoids division by the exactly-zero
        # imaginary variance of the terminal Fourier coordinate.
        model = FourierFlow(hidden=32, fft_size=25, n_flows=5, FFT=True, flip=True, normalize=False)
        train_j = np.concatenate([train[:, :, j], train[:, -1:, j]], axis=1)
        model.fit(
            train_j,
            epochs=args.epochs,
            batch_size=min(args.batch_size, len(train)),
            learning_rate=1e-3,
            display_step=max(1, args.epochs),
        )
        generated[:, :, j] = model.sample(args.num_samples)[:, :24]
        states.append(model.state_dict())
    torch.save(states, out / "models/model.pt")
    return generated, {
        "epochs_per_component": args.epochs,
        "n_component_flows": 32,
        "architecture": "official Fourier Flow",
    }


def train_aec_gan(train, test, args, device, out):
    # signatory is imported by an unrelated optional SigCWGAN module in the repository.
    sys.modules.setdefault("signatory", types.ModuleType("signatory"))
    sys.path.insert(0, str(args.vendor_dir / "aec_gan"))
    import lib.algos.asganadv as aec_module
    from lib.algos.asganadv import AECGAN
    from lib.algos.base import BaseConfig

    # Keep the indexed tensor and its indices on exactly the same device. The
    # upstream helper hard-codes `.cuda()` while its caller may retain x_real on
    # CPU, which fails on CUDA before the first optimization step.
    x = torch.from_numpy(train).float().to(device)
    aec_module.sample_indices = lambda n, b: torch.randperm(n, device=x.device)[:b]
    cfg = BaseConfig(
        seed=args.seed,
        batch_size=min(args.batch_size, len(train)),
        device=str(device),
        p=3,
        q=21,
        hidden_dims=(128, 128, 128),
        total_steps=args.steps,
        eps=0.01,
        experiment_directory=str(out),
        noise_type="min_adv",
        use_ec=2,
    )
    model = AECGAN(cfg, x)
    model.fit()
    past = torch.from_numpy(test[: args.num_samples, :3]).float().to(device)
    model.G.eval()
    with torch.no_grad():
        raw, corrected, _ = model.G.sample(21, past)
        generated = torch.cat([past, corrected], dim=1).cpu().numpy()
    torch.save(
        {
            "generator": model.G.state_dict(),
            "discriminator": model.D.state_dict(),
            "training_loss": dict(model.training_loss),
        },
        out / "models/model.pt",
    )
    return generated, {
        "steps": args.steps,
        "conditioning": "observed evaluation patches 1-3",
        "architecture": "AEC-GAN; generated array includes the three-patch observed prefix",
    }


def train_timevqvae(train, args, device, out):
    sys.path.insert(0, str(args.vendor_dir / "timevqvae" / "src"))
    from timevqvae.vqvae import VQVAE
    from timevqvae.maskgit import MaskGIT, PriorModelConfig

    vq = VQVAE(
        in_channels=32,
        input_length=24,
        n_fft=4,
        init_dim=4,
        hid_dim=64,
        downsampled_width_l=3,
        downsampled_width_h=6,
        codebook_size_l=128,
        codebook_size_h=128,
        kmeans_init=True,
        codebook_dim=8,
    ).to(device)
    dl = loader(train, args.batch_size, args.seed, channels_first=True)
    opt = torch.optim.Adam(vq.parameters(), lr=2e-4)
    hist1 = []
    vq.train()
    for epoch in range(args.vq_epochs):
        for (x,) in dl:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            o = vq(x)
            loss = (
                sum(o.recons_loss.values()) + o.vq_losses["LF"]["loss"] + o.vq_losses["HF"]["loss"]
            )
            loss.backward()
            opt.step()
        hist1.append(float(loss.detach()))
    pcfg = PriorModelConfig(
        hidden_dim=64,
        n_layers=3,
        heads=4,
        ff_mult=2,
        use_rmsnorm=True,
        p_unconditional=0.1,
        model_dropout=0.1,
        emb_dropout=0.1,
    )
    prior = MaskGIT(
        vqvae=vq,
        lf_choice_temperature=1.0,
        hf_choice_temperature=1.0,
        lf_num_sampling_steps=8,
        hf_num_sampling_steps=8,
        lf_codebook_size=128,
        hf_codebook_size=128,
        transformer_embedding_dim=64,
        lf_prior_model_config=pcfg,
        hf_prior_model_config=pcfg,
        classifier_free_guidance_scale=1.0,
        n_classes=1,
    ).to(device)
    po = torch.optim.AdamW(prior.parameters(), lr=2e-4)
    hist2 = []
    for epoch in range(args.prior_epochs):
        for (x,) in dl:
            x = x.to(device)
            cond = torch.zeros(len(x), 1, dtype=torch.long, device=device)
            po.zero_grad(set_to_none=True)
            loss = prior(x, cond).total_mask_prediction_loss
            loss.backward()
            po.step()
        hist2.append(float(loss.detach()))
    prior.eval()
    with torch.no_grad():
        tl, th = prior.iterative_decoding(args.num_samples, class_condition=0, device=device)
        generated = (
            (
                prior.decode_token_ind_to_timeseries(tl, "lf")
                + prior.decode_token_ind_to_timeseries(th, "hf")
            )
            .transpose(1, 2)
            .cpu()
            .numpy()
        )
    torch.save(
        {
            "vqvae": vq.state_dict(),
            "prior": prior.state_dict(),
            "vq_history": hist1,
            "prior_history": hist2,
        },
        out / "models/model.pt",
    )
    return generated, {
        "vq_epochs": args.vq_epochs,
        "prior_epochs": args.prior_epochs,
        "architecture": "official dual-frequency TimeVQVAE + MaskGIT",
    }
