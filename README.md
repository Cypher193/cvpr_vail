# Know vs Show — pilot

Pilot for *Know vs Show: Probing and Steering Physical Knowledge in Video World Models*.
It checks that the measurement pipeline works before the main run: tracking, scoring, probe signal,
gravity-slope fits, steering and budget. Gates G1–G10 are in [configs/gates.yaml](configs/gates.yaml);
commit them before any pilot run and do not edit thresholds after seeing data.

## Layout

| Path | What |
|---|---|
| `configs/pilot.yaml` | models, input window, render settings, experiments |
| `configs/gates.yaml` | go/no-go thresholds (pre-registered) |
| `pilot/scenes.py` | scene specs with exact analytic ground truth (constant velocity, bounces, two-ball ratios, gravity switches) |
| `pilot/render.py` | Blender renders of the specs (Poly Haven HDRIs and textures) |
| `pilot/models.py` | Cosmos-Predict2.5-2B, LTX-Video-2B (0.9.5), Wan2.1-VACE-1.3B wrappers with activation capture |
| `pilot/generate.py` | one experiment × one model → `runs/{exp}/{model}/{scene}_s{seed}/` |
| `pilot/track.py` | SAM2 tracking, metre conversion, kinematic fits; P0 validation on renders and QuantiPhy val |
| `pilot/probe.py` | ridge probes, scene-split CV, VAE / control-task / mean baselines |
| `pilot/metrics.py` | symmetric MRA, original MRA, SILog, γ, residual, CRPS, bootstrap, Holm (`python -m pilot.metrics` self-check) |
| `pilot/download.py` | Physics-IQ and QuantiPhy val clips, real-clip manifest with hand-drawn boxes |

## RunPod setup

Image: `ghcr.io/cypher193/cvpr_vail:latest`, built by GitHub Actions from the [Dockerfile](Dockerfile) on every
change to it or to `requirements.txt` (torch 2.8 / CUDA 12.8 base with SSH, all Python packages, Blender in `/opt/bpy`).
The image holds dependencies only; code and data live on the network volume.

Data lives on a private Hugging Face dataset repo (`CypherNotHere/cvpr-vail-pilot`, override with `HF_DATA_REPO`):
`setup_runpod.sh` pulls it and starts a background push every 15 min (`scripts/sync.py`). Pods are disposable,
so any datacenter and community cloud work. Model weights (~100 GB) are re-downloaded on each new pod.

Pod template:
- Container image: `ghcr.io/cypher193/cvpr_vail:latest` (if the package is private, add GHCR registry credentials
  in RunPod: username = GitHub user, password = a classic token with `read:packages`; or make the package public)
- Container disk 30 GB; volume disk 200 GB at `/workspace` (pod-attached; no network volume needed)
- Expose TCP 22 (SSH) and HTTP 8888 (Jupyter)
- Environment: `HF_TOKEN = {{ RUNPOD_SECRET_HF_TOKEN }}` (store the token as a RunPod secret)
- Filter hosts to CUDA 12.8 or newer
- One pod can do everything if its GPU has RT cores and >= 48 GB (RTX A6000, L40S); A100 80 GB also works.

Before terminating a pod: `python scripts/sync.py push`.

## Order of work

All steps run on RunPod (A100 80 GB, network volume >= 150 GB at /workspace).

1. `bash scripts/setup_runpod.sh` (needs `HF_TOKEN`): packages, model weights, real clips.
2. `bash scripts/render_all.sh`: builds the Blender venv and renders all scenes (~12,500 frames; Cycles on the A100).
3. `bash scripts/smoke_test.sh` (day 1).
   Tracker check (G1a): `python -m pilot.track validate data/renders/kmain --scenes data/scenes/kmain` and
   `python -m pilot.track qval`.
4. Generation, e.g. `python -m pilot.generate --model vace13 --exp b1` for each model × {b1, b2nl, b5, bratio, b1real}.
5. Tracking of generated clips, probes (`python -m pilot.probe --model vace13 --exps b1 b5 --target g`), gates report.

## Design notes found while building

- **Input window is about 0.3 s, not 1 s.** Cosmos-Predict2.5 (diffusers) accepts at most 2 latent conditioning
  frames = 5 pixel frames = 0.3125 s at 16 fps. All models get the same duration (LTX: 9 frames at 24 fps = 0.375 s).
- **QuantiPhy val clips are 1.3–3.3 s long**, so they serve as the real-footage tracker check (9 kinematic
  questions on 3 clips) and as real inputs, not as a main scene source.
- **Physics-IQ:** only 4 of 10 candidate scenarios keep the object in view after the input window
  (ball-and-block-fall, ball-ramp, balls-collide, mirror-ball-fall); centre views are used.
- **Bounce fits use one model fit** (start height, start velocity, g, restitution) over the whole track,
  because separate per-flight parabolas fail at 16 fps.
- Probe scores have a high floor (predicting the mean already scores ~0.68 when most g = 9.81), so read
  probe results only relative to the mean, VAE and control-task baselines.

Not yet written (needed from day 6): `invert.py` (C1-inv / C1-noise), `likelihood.py` (C2-g), `steer.py` (C9/C7),
random-init baseline, gates report.
