"""P11: time one Cosmos 3 Nano clip (model card's diffusers example, guardrail on). Runs in /workspace/venv-nano.
Writes runs/timing/cosmos3_nano/<scene>_s0/{video.mp4, meta.json}."""
import json
import time
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

spec = json.loads(Path("data/scenes/kmain/kmain_cv_00.json").read_text(encoding="utf-8"))
t0 = time.time()
pipe = Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda",
                                           enable_safety_checker=True)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)
load_s = time.time() - t0
torch.cuda.reset_peak_memory_stats()
t1 = time.time()
res = pipe(prompt=spec["caption"], num_frames=93, height=480, width=832, num_inference_steps=35, guidance_scale=6.0,
           generator=torch.Generator(device="cuda").manual_seed(0))
wall = time.time() - t1
out = Path("runs/timing/cosmos3_nano") / f"{spec['id']}_s0"
out.mkdir(parents=True, exist_ok=True)
export_to_video(res.video, str(out / "video.mp4"), fps=16)
meta = dict(model="cosmos3_nano", scene=spec["id"], seed=0, wall_s=round(wall, 1), load_s=round(load_s, 1),
            peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), height=480, width=832, num_frames=93, steps=35,
            note="text-to-video timing only (480p, 93 frames, 35 steps), guardrail enabled")
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print(meta)
