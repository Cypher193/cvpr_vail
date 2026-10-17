"""P7: C1-inv vs C1-noise. Activations from a REAL clip (full video, on purpose, as in Invisible Hand / RigidBench),
obtained two ways at the same noise levels the generation runs capture:

  noise:  x_s = (1 - s) x0 + s * eps, one forward pass                      (RigidBench-style)
  inv:    integrate the flow ODE from x0 (s = 0) up to s, then one forward pass (inversion, Invisible Hand-style)

    python -m pilot.invert --model ltx2b --scenes 30
    python -m pilot.invert --model wan22_5b --scenes 30

Writes runs/p7/{model}_{noise|inv}/{scene}_s0/acts.npz, probed later with pilot.probe (target hspeed).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from pilot.models import Capture, _pil, token_grid
from pilot.scenes import object_track, project
from pilot.track import load_video


class Adapter:
    """Model-specific pieces: encode a clip to normalised latents, text embeddings, velocity at noise level s."""

    def __init__(self, name, cfg, pipe, device, dtype):
        self.name, self.cfg, self.pipe, self.device, self.dtype = name, cfg, pipe, device, dtype
        self.ltx = name.startswith("ltx")

    @torch.no_grad()
    def encode(self, frames):
        c, vae = self.cfg, self.pipe.vae
        x = self.pipe.video_processor.preprocess_video(_pil(frames, c["width"], c["height"]), c["height"], c["width"])
        lat = vae.encode(x.to(self.device, vae.dtype)).latent_dist.mean
        if self.ltx:
            return self.pipe._normalize_latents(lat, vae.latents_mean, vae.latents_std, vae.config.scaling_factor)
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(lat)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(lat)
        return (lat - mean) / std                           # inverse of the pipeline's decode normalisation

    @torch.no_grad()
    def text(self, prompt):
        out = self.pipe.encode_prompt(prompt, do_classifier_free_guidance=False, device=self.device)
        return (out[0], out[1]) if self.ltx else (out[0], None)

    @torch.no_grad()
    def velocity(self, x, s, text):
        emb, mask = text
        t = torch.full((x.shape[0],), s * 1000.0, device=self.device)
        tr = self.pipe.transformer
        if self.ltx:
            B, C, F, H, W = x.shape
            p, pt = tr.config.patch_size, tr.config.patch_size_t
            v = tr(hidden_states=self.pipe._pack_latents(x, p, pt).to(self.dtype), encoder_hidden_states=emb,
                   timestep=t, encoder_attention_mask=mask, num_frames=F, height=H, width=W,
                   rope_interpolation_scale=(self.pipe.vae_temporal_compression_ratio / self.cfg["fps"],
                                             self.pipe.vae_spatial_compression_ratio,
                                             self.pipe.vae_spatial_compression_ratio),
                   return_dict=False)[0]
            return self.pipe._unpack_latents(v.float(), F, H, W, p, pt)
        if getattr(self.pipe.config, "expand_timesteps", False):   # Wan2.2 TI2V: one timestep per token
            _, _, F, H, W = x.shape
            ph, pw = tr.config.patch_size[1:]
            t = t[:, None].expand(-1, F * (H // ph) * (W // pw))
        return tr(hidden_states=x.to(self.dtype), timestep=t, encoder_hidden_states=emb, return_dict=False)[0].float()


def noise_levels(pipe, steps, fractions):
    """The sigmas at the denoising steps the generation runs capture (same fractions, same scheduler)."""
    pipe.scheduler.set_timesteps(steps)
    sig = pipe.scheduler.sigmas.cpu().numpy()
    return [float(sig[int(round(f * (steps - 1)))]) for f in fractions]


def object_mask(cfg, grid, spec, n_frames, fps):
    """Object tokens in every latent frame, from the exact track at that frame's centre time."""
    F, H, W = grid
    vt = cfg["vae_stride"][0]
    cam = spec["camera"]
    sx, sy = W / cam["width"], H / cam["height"]
    m = np.zeros(grid, bool)
    for k in range(F):
        i = 0 if k == 0 else min(vt * k - (vt - 1) // 2, n_frames - 1)   # centre pixel frame of latent frame k
        for o in spec["objects"]:
            x, z = object_track(spec, o, np.array([i / fps]))
            u, v = project(cam, x, z)
            r = o["size_m"] / 2 * cam["fx"] / cam["distance"]
            m[k, max(int((v[0] - r) * sy), 0):int(np.ceil((v[0] + r) * sy)),
              max(int((u[0] - r) * sx), 0):int(np.ceil((u[0] + r) * sx))] = True
    return m


@torch.no_grad()
def run_clip(ad, frames, spec, levels, n_ode=10, seed=0):
    """-> {'noise': arrays, 'inv': arrays} pooled block activations at each noise level."""
    c = ad.cfg
    grid = token_grid(c)
    mask = object_mask(c, grid, spec, len(frames), c["fps"])
    x0 = ad.encode(frames)
    text = ad.text(spec["caption"])
    gen = torch.Generator(ad.device).manual_seed(seed)
    out = {}
    for method in ("noise", "inv"):
        # batch of 1, no guidance: the cond pass is batch item 0 for every model here
        cap = Capture(dict(c, cfg="separate"), len(levels), [i / max(len(levels) - 1, 1) for i in range(len(levels))],
                      grid, mask)
        cap.attach(ad.pipe.transformer)
        x, s_prev = x0.clone(), 0.0
        for j, s in enumerate(levels if method == "noise" else sorted(levels)):
            if method == "noise":
                eps = torch.randn(x0.shape, generator=gen, device=ad.device, dtype=x0.dtype)
                x = (1 - s) * x0 + s * eps
            else:                                             # Euler steps of the flow ODE from s_prev to s
                cap.step = -1                                 # not captured while integrating
                for a, b in zip(np.linspace(s_prev, s, n_ode + 1)[:-1], np.linspace(s_prev, s, n_ode + 1)[1:]):
                    x = x + (b - a) * ad.velocity(x, float(a), text)
                s_prev = s
            cap.step = cap.steps[levels.index(s)]
            ad.velocity(x, s, text)
        cap.detach()
        out[method] = cap.arrays()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--config", default="configs/pilot.yaml")
    a = ap.parse_args()
    import diffusers
    cfg = yaml.safe_load(open(a.config))
    mcfg = cfg["models"][a.model]
    cls = "LTXPipeline" if a.model.startswith("ltx") else mcfg["pipeline"]
    kw = dict(torch_dtype=torch.bfloat16) | ({"revision": mcfg["revision"]} if mcfg.get("revision") else {})
    pipe = getattr(diffusers, cls).from_pretrained(mcfg["repo"], **kw).to("cuda")
    ad = Adapter(a.model, mcfg, pipe, "cuda", torch.bfloat16)
    levels = noise_levels(pipe, mcfg.get("steps", 50), cfg["capture_fractions"])
    scenes = sorted(Path(cfg["paths"]["data"], "scenes", "kmain").glob("*.json"))[: a.scenes]
    for f in scenes:
        spec = json.loads(f.read_text(encoding="utf-8"))
        outs = {m: Path(cfg["paths"]["runs"]) / "p7" / f"{a.model}_{m}" / f"{spec['id']}_s0" for m in ("noise", "inv")}
        if all((o / "acts.npz").exists() for o in outs.values()):
            continue
        video, fps = load_video(Path(cfg["paths"]["data"], "renders", "kmain", f"{spec['id']}.mp4"))
        frames = list(video[: mcfg["num_frames"]])
        t0 = time.time()
        res = run_clip(ad, frames, spec, levels)
        for m, o in outs.items():
            o.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(o / "acts.npz", **res[m])
            (o / "meta.json").write_text(json.dumps(dict(model=f"{a.model}_{m}", scene=spec["id"], seed=0,
                                                         levels=levels, wall_s=round(time.time() - t0, 1))))
        print(f"p7 {a.model} {spec['id']}: {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
