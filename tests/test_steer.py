"""CPU check: the steering hook changes the output, and alpha = 0 leaves it identical.  python -m tests.test_steer"""
import numpy as np
import torch

from pilot.models import Generator, object_token_mask, token_grid
from pilot.steer import steer_hook
from tests import tiny

RUN = {"capture_fractions": [0.25, 0.5, 0.75]}
for name, build in tiny.BUILDERS.items():
    pipe, cfg = build()
    gen = Generator(name, cfg, RUN, device="cpu", pipe=pipe, dtype=torch.float32)
    fr, box = tiny.frames(cfg["cond_frames"]), [10, 10, 30, 30]
    grid = token_grid(cfg)
    idx = torch.from_numpy(np.flatnonzero(object_token_mask(cfg, grid, box, (64, 48))))
    d = np.random.default_rng(0).normal(size=gen.blocks()[0].__dict__.get("dim", 0) or 1)
    D = gen.generate(fr, "a ball", 0, box, (64, 48))["acts"][f"{cfg['blocks'][0]}/all"].shape[-1]
    w = np.random.default_rng(1).normal(size=D).astype(np.float32); w /= np.linalg.norm(w)
    run = lambda alpha: gen.generate(fr, "a ball", 0, box, (64, 48), capture=False,
                                     hooks=[(gen.blocks()[0], steer_hook(w, 1.0, idx, alpha))])["video"].astype(int)
    base = gen.generate(fr, "a ball", 0, box, (64, 48), capture=False)["video"].astype(int)
    zero, strong = run(0.0), run(50.0)
    print(name, "alpha=0 diff", np.abs(zero - base).max(), "| alpha=50 diff", np.abs(strong - base).mean().round(2))
    assert np.abs(zero - base).max() == 0 and np.abs(strong - base).mean() > 0.05, name   # any effect; tiny random weights
print("steering checks passed")
