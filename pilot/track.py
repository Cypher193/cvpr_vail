"""Show readout: SAM2 masks -> centroids -> metres -> kinematic fits.

    python -m pilot.track validate data/renders/kmain --scenes data/scenes/kmain   # P0 on renders
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pilot import metrics

_SAM = {}


def load_video(path):
    """-> uint8 frames (T, H, W, 3) RGB, fps."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[..., ::-1])
    return np.stack(frames), float(fps)


def sam2_masks(frames, boxes, model_id="facebook/sam2.1-hiera-large", device=None, prompt_frame=0):
    """frames uint8 (T, H, W, 3); boxes [[x0, y0, x1, y1], ...] on prompt_frame. Returns bool (n_obj, T, H, W).
    Propagates forward from the prompt frame, then backward if it is not the first frame."""
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if model_id not in _SAM:
        _SAM[model_id] = (Sam2VideoProcessor.from_pretrained(model_id),
                          Sam2VideoModel.from_pretrained(model_id).to(device, dtype=dtype).eval())
    proc, model = _SAM[model_id]
    T, H, W = frames.shape[:3]
    sess = proc.init_video_session(video=list(frames), inference_device=device, dtype=dtype)
    proc.add_inputs_to_inference_session(sess, frame_idx=prompt_frame, obj_ids=list(range(1, len(boxes) + 1)),
                                         input_boxes=[[list(map(float, b)) for b in boxes]])
    out = np.zeros((len(boxes), T, H, W), bool)
    with torch.inference_mode():
        model(inference_session=sess, frame_idx=prompt_frame)   # segment the prompted frame first
        for reverse in ([False, True] if prompt_frame > 0 else [False]):
            for o in model.propagate_in_video_iterator(sess, start_frame_idx=prompt_frame, reverse=reverse):
                m = proc.post_process_masks([o.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
                out[:, o.frame_idx] = m.reshape(len(boxes), H, W).cpu().numpy()
    return out


def centroids(masks):
    """masks (T, H, W) -> centroid (T, 2) in pixels (nan where empty), area (T,)."""
    T = masks.shape[0]
    c = np.full((T, 2), np.nan)
    area = masks.reshape(T, -1).sum(1).astype(float)
    for i in np.flatnonzero(area > 0):
        v, u = np.nonzero(masks[i])
        c[i] = u.mean(), v.mean()
    return c, area


def px_per_m(area_cond, size_m, shape="sphere"):
    """Scale from the object's known size, measured on the input frames only."""
    a = np.median(area_cond[area_cond > 0])
    width_px = 2 * np.sqrt(a / np.pi) if shape == "sphere" else np.sqrt(a)
    return width_px / size_m


def measurable(area, i0, min_present=0.8, area_lo=0.5, area_hi=2.0):
    """Object present in >= 80% of scored frames with area within 0.5-2x its frame-0 area."""
    ref, a = area[0], area[i0:]
    if ref <= 0 or len(a) == 0:
        return False
    ok = (a > 0) & (a >= area_lo * ref) & (a <= area_hi * ref)
    return bool(ok.mean() >= min_present)


# ---------------------------------------------------------------- fits (metres, z up, t in seconds)
def flight_segments(t, z, min_pts=5):
    """Split a bouncing trajectory at floor contacts; fit z = c0 + c1 t + c2 t^2 per flight.
    Contact samples are excluded, so no fit spans a bounce. Resting stretches are skipped."""
    ok = ~np.isnan(z)
    t, z = t[ok], z[ok]
    if len(z) < min_pts:
        return []
    span = np.ptp(z)
    low = z.min() + 0.15 * span
    dz = np.diff(z)
    contacts = [i + 1 for i in range(len(dz) - 1) if dz[i] < 0 <= dz[i + 1] and z[i + 1] < low]
    edges = [0] + contacts + [len(z)]
    segs = []
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        idx = np.arange(a if k == 0 else a + 1, b)          # drop the contact sample that starts a flight
        if len(idx) >= min_pts and np.ptp(z[idx]) > 0.05 * span:
            c2, c1, c0 = np.polyfit(t[idx], z[idx], 2)
            if c2 < 0:                                      # a free flight is concave down
                segs.append(dict(t0=t[idx[0]], t1=t[idx[-1]], c0=c0, c1=c1, c2=c2, n=len(idx)))
    return segs


def speed_at(seg, z_floor):
    """|dz/dt| where a flight parabola is at height z_floor: (dz/dt)^2 = c1^2 + 4 c2 (z - c0)."""
    return float(np.sqrt(max(seg["c1"] ** 2 + 4 * seg["c2"] * (z_floor - seg["c0"]), 0.0)))


def fit_bounce(t, z, z_floor, g_grid=np.geomspace(0.3, 60, 25), e_grid=(0.3, 0.5, 0.7, 0.85, 0.95)):
    """Fit the bouncing-ball law (start height, start velocity, g, restitution e) to a whole track.
    Works at low frame rates because it uses every sample, not separate per-flight parabolas.
    Returns g, e, rmse (m) and the fitted segments; g is in the clip's own clock."""
    from scipy.optimize import least_squares
    from pilot.scenes import eval_segments, vertical_segments
    ok = ~np.isnan(z)
    t, z = t[ok] - t[ok][0], z[ok]
    if len(z) < 6:
        return None
    T = t[-1] + 1e-6

    def model(p):
        segs, _ = vertical_segments(max(p[0], z_floor), p[1], [(0.0, abs(p[2]))], np.clip(p[3], 0, 0.99), z_floor, T)
        return eval_segments(segs, t)

    v0 = (z[1] - z[0]) / (t[1] - t[0])
    g_seeds = [-2 * s["c2"] for s in flight_segments(t, z)]           # per-flight estimates as extra starts
    starts = sorted(((z[0], v0, g, e) for g in [*g_grid, *g_seeds] for e in e_grid),
                    key=lambda p: np.sum((model(p) - z) ** 2))[:3]
    fits = [least_squares(lambda p: model(p) - z, p0, x_scale=[0.1, 0.5, p0[2], 0.1],
                          bounds=([z_floor, -20, 0.05, 0.0], [z_floor + 10, 20, 200, 0.99])) for p0 in starts]
    r = min(fits, key=lambda f: f.cost)
    p = r.x
    segs, contacts = vertical_segments(p[0], p[1], [(0.0, p[2])], p[3], z_floor, T)
    return dict(g=float(p[2]), e=float(p[3]), z0=float(p[0]), vz0=float(p[1]),
                rmse=float(np.sqrt(np.mean(r.fun ** 2))), contacts=contacts)


def readout(family, t, xs, zs, t_start):
    """Kinematic readout from tracks in metres. xs, zs: lists per object. Only t >= t_start is scored."""
    w = t >= t_start
    out = {}
    if family == "cv":
        x, z = xs[0][w], zs[0][w]
        ok = ~np.isnan(x)
        c = np.polyfit(t[w][ok], x[ok], 2)
        out.update(v=abs(np.polyval(np.polyder(c), t[w][ok]).mean()), a=2 * c[0],
                   residual=metrics.accel_residual(np.c_[x[ok], z[ok]], t[1] - t[0]))
        return out
    if family == "bounce":
        x, z = xs[0][w], zs[0][w]
        ok = ~np.isnan(x)
        out["vx"] = abs(np.polyfit(t[w][ok], x[ok], 1)[0])
        fit = fit_bounce(t[w], z, np.nanmin(zs[0]))
        if fit:
            out.update(g=fit["g"], e=fit["e"], fit_rmse=fit["rmse"])
        segs = flight_segments(t[w], z)
        if segs:
            longest = max(segs, key=lambda s: s["n"])
            sel = (t[w] >= longest["t0"]) & (t[w] <= longest["t1"]) & ok
            out["residual"] = metrics.accel_residual(z[sel], t[1] - t[0])
        return out
    if family == "ratio":
        flights = []
        for z in zs:
            z_floor = np.nanmin(z)
            fit = fit_bounce(t[w], z[w], z_floor)
            if not fit or not fit["contacts"]:
                return out
            t_land = t[w][0] + fit["contacts"][0]           # launch at t = 0, so flight time = first landing
            v_up = np.sqrt(2 * fit["g"] * max(fit["z0"] - z_floor, 0) + fit["vz0"] ** 2)
            flights.append((t_land, v_up ** 2 / (2 * fit["g"])))
        (T1, h1), (T2, h2) = flights
        out.update(t_ratio=T1 / T2, h_ratio=h1 / h2)
        return out
    raise ValueError(family)


def gt_boxes(spec, frame=0):
    """Ground-truth boxes on one frame from the analytic tracks."""
    boxes = []
    for o, tr in zip(spec["objects"], spec["tracks"]):
        u, v, r = tr["u_px"][frame], tr["v_px"][frame], tr["radius_px"]
        boxes.append([u - r, v - r, u + r, v + r])
    return boxes


def track_clip(video_path, spec, boxes=None, t_start=0.0, cond_frames=1):
    """Track a clip of a known scene. Returns readout dict plus measurability and pixel tracks."""
    frames, fps = load_video(video_path)
    t = np.arange(len(frames)) / fps
    masks = sam2_masks(frames, boxes or gt_boxes(spec))
    i0 = int(np.searchsorted(t, t_start))
    xs, zs, ok = [], [], True
    for o, m in zip(spec["objects"], masks):
        c, area = centroids(m)
        s = px_per_m(area[:cond_frames], o["size_m"], o["shape"])
        cam = spec["camera"]
        xs.append((c[:, 0] - cam["cx"]) / s)
        zs.append(-(c[:, 1] - cam["cy"]) / s)
        ok &= measurable(area, i0)
    res = dict(measurable=ok, fps=fps, n_frames=len(frames))
    if ok:
        res.update(readout(spec["family"], t, xs, zs, t_start))
    res["track_px"] = [np.c_[(x * 1), (z * 1)].round(4).tolist() for x, z in zip(xs, zs)]
    return res


def validate(render_dir, scene_dir, out_csv):
    """P0 on renders: tracker readout vs exact ground truth (gate G1a)."""
    import pandas as pd
    rows = []
    for vid in sorted(v for v in Path(render_dir).glob("*.mp4") if v.stem.count("_") == 2):   # full clips only
        spec = json.loads((Path(scene_dir) / f"{vid.stem}.json").read_text(encoding="utf-8"))
        r = track_clip(vid, spec, cond_frames=5)
        row = dict(scene=vid.stem, family=spec["family"], measurable=r["measurable"])
        for k, v in spec["gt"].items():
            if k in r:
                row[f"{k}_gt"], row[f"{k}_tr"] = v, r[k]
                row[f"{k}_sym_mra"] = float(metrics.sym_mra(r[k], v))
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    mra = df.filter(like="_sym_mra").stack().mean()
    print(f"P0 renders: mean symmetric MRA {mra:.3f}, measurable {df.measurable.mean():.2f}")
    return df


def local_kinematics(t, xy, t_q, half_window=0.125):
    """Speed and acceleration magnitude at t_q from a local quadratic fit of a 2D track (metres)."""
    sel = (np.abs(t - t_q) <= half_window) & ~np.isnan(xy[:, 0])
    if sel.sum() < 4:
        return np.nan, np.nan
    cx, cy = (np.polyfit(t[sel] - t_q, xy[sel, k], 2) for k in (0, 1))
    return float(np.hypot(cx[1], cy[1])), float(np.hypot(2 * cx[0], 2 * cy[0]))


def validate_qval(qfile="data/real/qval_questions.json", root="data/quantiphy_val/validation_videos",
                  out_csv="results/p0_qval.csv"):
    """P0 on real footage: QuantiPhy val answers for 2D-motion clips recovered by the tracker (gate G1a).
    Each object has its own box, prompt time and known size (the scale comes from the object itself)."""
    import pandas as pd
    rows = []
    for clip, q in json.loads(Path(qfile).read_text(encoding="utf-8")).items():
        frames, fps = load_video(Path(root) / f"{clip}.mp4")
        t = np.arange(len(frames)) / fps
        tracks = {}
        for name, o in q["objects"].items():
            f0 = int(round(o["prompt_t"] * fps))
            m = sam2_masks(frames, [o["box"]], prompt_frame=f0)[0]
            c, area = centroids(m)
            near = area[max(f0 - 3, 0):f0 + 4]                    # size measured around the prompt frame
            tracks[name] = c / px_per_m(near, o["size_m"])
        for obj, quantity, tq, unit, answer in q["questions"]:
            if quantity == "avg_speed":                          # tq = (t0, t1): mean displacement rate
                i0, i1 = (int(round(x * fps)) for x in tq)
                pred = np.linalg.norm(tracks[obj][i1] - tracks[obj][i0]) / (t[i1] - t[i0]) / unit
            else:
                v, acc = local_kinematics(t, tracks[obj], tq, half_window=0.125 if quantity == "speed" else 0.25)
                pred = (v if quantity == "speed" else acc) / unit
            rows.append(dict(clip=clip, object=obj, quantity=quantity, t=str(tq), answer=answer, tracked=pred,
                             sym_mra=float(metrics.sym_mra(pred, answer)) if np.isfinite(pred) else 0.0))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"P0 QuantiPhy val: mean symmetric MRA {df.sym_mra.mean():.3f} over {len(df)} questions")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("renders")
    v.add_argument("--scenes", required=True)
    v.add_argument("--out", default="results/p0_renders.csv")
    q = sub.add_parser("qval")
    q.add_argument("--out", default="results/p0_qval.csv")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.cmd == "validate":
        validate(a.renders, a.scenes, a.out)
    else:
        validate_qval(out_csv=a.out)
