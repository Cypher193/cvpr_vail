"""CPU check of P8 on tiny Cosmos: losses finite per k, implied multiplier inside the grid.  python -m tests.test_likelihood"""
import numpy as np
import torch

from pilot.likelihood import implied_multiplier, run
from tests import tiny

assert abs(implied_multiplier(np.array([0.25, 0.5, 1, 2, 4]), (np.log([0.25, 0.5, 1, 2, 4]) - np.log(1.5)) ** 2) - 1.5) < 1e-6
pipe, cfg = tiny.cosmos()
run_cfg = {"paths": {"data": "data"}}
df, summ = run(pipe, cfg, run_cfg, "cpu", torch.float32, n_scenes=1, n_sigmas=2, n_noise=1,
               frames_fn=lambda spec: tiny.frames(cfg["num_frames"], seed=int(spec["gt"]["cont_mult"] * 4)))
print(df.round(5).to_string(index=False)); print(summ.round(3).to_string(index=False))
assert len(df) == 7 and np.isfinite(df.loss).all() and 0.25 <= summ.k_implied.iloc[0] <= 4
print("likelihood checks passed")
