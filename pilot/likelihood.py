"""P8 / C2-g: likelihood-implied gravity (Cosmos-2B). Which continuation does the model find most likely?

Each kcont scene keeps its first 5 frames (the Cosmos input window) at Earth gravity, then continues under
k x g, k in {0.25 ... 4}. For each k: denoising loss ||v_pred - (eps - x0)||^2 on the continuation latent
frames, averaged over the scheduler's noise levels and fixed noise draws (identical across k). The implied
multiplier is the minimum of a parabola fitted to loss vs ln k; implied gravity = 9.81 * k*.

    python -m pilot.likelihood --scenes 5

Writes results/p8_cosmos2b.csv (per scene and k) and results/p8_summary.csv (k*, symmetric MRA vs 1).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from pilot import metrics
from pilot.models import _pil
from pilot.track import load_video


class CosmosLoss:
    def __init__(self, pipe, cfg, device, dtype):
        self.pipe, self.cfg, self.device, self.dtype = pipe, cfg, device, dtype

    @torch.no_grad()
    def latents(self, frames):
        c, p = self.cfg, self.pipe
        x = p.video_processor.preprocess_video(_pil(frames, c["width"], c["height"]), c["height"], c["width"])
        z = p.vae.encode(x.to(self.device, p.vae.dtype)).latent_dist.mode().float()
        return (z - p.latents_mean.to(z)) * p.latents_std.to(z)       # pipeline: latents_std holds 1/std

    @torch.no_grad()
    def text(self, prompt):
        if not self.pipe.safety_checker.check_text_safety(prompt):    # NVIDIA guardrail stays in the loop
            raise ValueError("prompt rejected by the Cosmos guardrail")
        return self.pipe.encode_prompt(prompt, do_classifier_free_guidance=False, device=self.device)[0]

    @torch.no_grad()
    def loss(self, x0, emb, sigmas, noises, n_cond_latent, cond_t=1e-4):
        B, C, T, H, W = x0.shape
        ind = torch.zeros(B, 1, T, 1, 1, device=self.device)
        ind[:, :, :n_cond_latent] = 1.0
        mask = ind.expand(-1, -1, -1, H, W)
        pad = x0.new_zeros(1, 1, self.cfg["height"], self.cfg["width"])
        out = []
        for s in sigmas:
            for eps in noises:
                xs = mask * x0 + (1 - mask) * ((1 - s) * x0 + s * eps)
                ts = ind * cond_t + (1 - ind) * s
                v = self.pipe.transformer(hidden_states=xs.to(self.dtype), condition_mask=mask.to(self.dtype),
                                          timestep=ts.to(self.dtype), encoder_hidden_states=emb,
                                          padding_mask=pad.to(self.dtype), return_dict=False)[0].float()
                err = (v - (eps - x0)) ** 2
                out.append(err[:, :, n_cond_latent:].mean().item())   # continuation frames only
        return float(np.mean(out))


def implied_multiplier(ks, losses):
    """Minimum of a parabola in ln k through the losses (falls back to the best grid point at the edges)."""
    x, y = np.log(ks), np.asarray(losses)
    a, b, _ = np.polyfit(x, y, 2)
    if a > 0:
        xm = -b / (2 * a)
        if x.min() <= xm <= x.max():
            return float(np.exp(xm))
    return float(ks[int(np.argmin(y))])


def run(pipe, cfg, run_cfg, device, dtype, n_scenes, n_sigmas=8, n_noise=4, scene_root=None, render_root=None,
        frames_fn=None):
    lo = CosmosLoss(pipe, cfg, device, dtype)
    scene_root = Path(scene_root or Path(run_cfg["paths"]["data"]) / "scenes" / "kcont")
    render_root = Path(render_root or Path(run_cfg["paths"]["data"]) / "renders" / "kcont")
    pipe.scheduler.set_timesteps(cfg["steps"])
    all_sig = pipe.scheduler.sigmas.cpu().numpy()[:-1]
    sigmas = [float(s) for s in all_sig[np.linspace(0, len(all_sig) - 1, n_sigmas).astype(int)]]
    n_cond_latent = 1 + (cfg["cond_frames"] - 1) // cfg["vae_stride"][0]
    specs = sorted(scene_root.glob("*.json"))
    bases = sorted({f.stem.rsplit("_x", 1)[0] for f in specs})[:n_scenes]
    rows = []
    for base in bases:
        variants = sorted((json.loads(f.read_text(encoding="utf-8")) for f in specs if f.stem.startswith(base + "_x")),
                          key=lambda s: s["gt"]["cont_mult"])
        emb = lo.text(variants[0]["caption"])
        gen = torch.Generator(device).manual_seed(0)
        noises = None
        for spec in variants:
            frames = frames_fn(spec) if frames_fn else list(load_video(render_root / f"{spec['id']}.mp4")[0])
            x0 = lo.latents(frames[: cfg["num_frames"]])
            if noises is None:                                  # same draws for every k
                noises = [torch.randn(x0.shape, generator=gen, device=device) for _ in range(n_noise)]
            rows.append(dict(base=base, k=spec["gt"]["cont_mult"],
                             loss=lo.loss(x0, emb, sigmas, noises, n_cond_latent)))
            print(f"p8 {spec['id']}: loss {rows[-1]['loss']:.5f}", flush=True)
    df = pd.DataFrame(rows)
    summ = []
    for base, g in df.groupby("base"):
        k = implied_multiplier(g.k.to_numpy(), g.loss.to_numpy())
        summ.append(dict(base=base, k_implied=k, g_implied=9.81 * k, sym_mra=float(metrics.sym_mra(k, 1.0))))
    return df, pd.DataFrame(summ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=5)
    ap.add_argument("--config", default="configs/pilot.yaml")
    a = ap.parse_args()
    import diffusers
    run_cfg = yaml.safe_load(open(a.config))
    cfg = run_cfg["models"]["cosmos2b"]
    pipe = diffusers.Cosmos2_5_PredictBasePipeline.from_pretrained(cfg["repo"], revision=cfg["revision"],
                                                                   torch_dtype=torch.bfloat16).to("cuda")
    df, summ = run(pipe, cfg, run_cfg, "cuda", torch.bfloat16, a.scenes)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/p8_cosmos2b.csv", index=False)
    summ.to_csv("results/p8_summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
