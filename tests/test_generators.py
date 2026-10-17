"""CPU check of generation + activation capture on tiny pipelines.  python -m tests.test_generators"""
import numpy as np
import torch

from pilot.models import Generator
from tests import tiny

RUN = {"capture_fractions": [0.25, 0.5, 0.75]}

for name, build in tiny.BUILDERS.items():
    pipe, cfg = build()
    gen = Generator(name, cfg, RUN, device="cpu", pipe=pipe, dtype=torch.float32)
    r = gen.generate(tiny.frames(cfg["cond_frames"]), "a ball", 0, obj_box=[10, 10, 30, 30], in_wh=(64, 48))
    shapes = {k: v.shape for k, v in r["acts"].items()}
    print(name, r["video"].shape, r["video"].dtype, shapes, {k: v.shape for k, v in r["vae"].items()},
          None if r["final_latent"] is None else tuple(r["final_latent"].shape))
    group = cfg["blocks"][0]
    assert f"{group}/obj" in r["acts"] and f"{group}/all" in r["acts"], name
    assert r["acts"][f"{group}/obj"].shape[1] == 3 and r["final_latent"] is not None
    assert r["video"].dtype == np.uint8 and r["video"].ndim == 4
    t2 = gen.generate([], "a ball", 0)                       # text only (B2-nl)
    assert t2["video"].ndim == 4
print("generator checks passed")
