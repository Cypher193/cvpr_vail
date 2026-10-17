"""CPU check of P7 (noising and inversion) on tiny LTX and Wan2.2-style pipelines.  python -m tests.test_invert"""
import json

import torch

from pilot.invert import Adapter, noise_levels, run_clip
from tests import tiny

spec = json.load(open("data/scenes/kmain/kmain_ca_00.json", encoding="utf-8"))
for name, build in (("ltx2b", tiny.ltx_t2v), ("wan22_5b", tiny.wan22)):
    pipe, cfg = build()
    ad = Adapter(name, cfg, pipe, "cpu", torch.float32)
    levels = noise_levels(pipe, cfg["steps"], [0.25, 0.5, 0.75])
    out = run_clip(ad, tiny.frames(cfg["num_frames"]), spec, levels, n_ode=3)
    g = cfg["blocks"][0]
    print(name, "levels", [round(x, 3) for x in levels], {m: {k: v.shape for k, v in a.items()} for m, a in out.items()})
    for m in ("noise", "inv"):
        assert out[m][f"{g}/obj"].shape[1] == 3 and out[m][f"{g}/all"].shape[1] == 3, (name, m)
    diff = abs(out["noise"][f"{g}/all"].astype(float) - out["inv"][f"{g}/all"].astype(float)).mean()
    assert diff > 0, "noise and inversion activations should differ"
print("invert checks passed")
