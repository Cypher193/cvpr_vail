"""Download real clips and write data/real/manifest.json (inputs for b1real) and the QuantiPhy tracker checks.

    python -m pilot.download

Boxes were drawn by hand on the first input frame (pixels of the source video).
Physics-IQ: centre view, take 1, 30 fps; only scenarios where the object stays in view after the input window.
QuantiPhy val: only clips whose kinematic questions are about a visible rigid object with a size prior.
"""
import json
from pathlib import Path
from urllib.request import urlretrieve

PIQ_URL = "https://storage.googleapis.com/physics-iq-benchmark/full-videos/take-1/30FPS/{}.mp4"
PIQ = {  # name: (file stem, box on the frame at 2.6875 s, caption without quantities)
    "ball-and-block-fall": ("0002_full-videos_30FPS_perspective-center_take-1_trimmed-ball-and-block-fall",
                            [880, 170, 1230, 555], "A tennis ball held by a clamp above two pillows. Static camera."),
    "ball-ramp": ("0020_full-videos_30FPS_perspective-center_take-1_trimmed-ball-ramp",
                  [115, 700, 385, 945], "A tennis ball rolls across a table towards a cardboard ramp. Static camera."),
    "balls-collide": ("0032_full-videos_30FPS_perspective-center_take-1_trimmed-balls-collide",
                      [295, 775, 555, 1010], "A blue ball and a yellow ball roll towards each other on a table. Static camera."),
    "mirror-ball-fall": ("0116_full-videos_30FPS_perspective-center_take-1_trimmed-mirror-ball-fall",
                         [1710, 1150, 2010, 1450], "An orange ball hangs from a magnet next to a mirror. Static camera."),
}
PIQ_INPUT_END_S = 3.0          # Physics-IQ conditioning ends at 3 s; the continuation is the ground truth

QVAL = {  # clip: (object boxes on frame 0, size prior in metres, questions: (object, quantity, time s, unit to SI))
    "captured_0021x": ({"ping pong ball": [474, 267, 492, 285]}, ("ping pong ball", 0.040),
                       [("ping pong ball", "speed", 1.2, 0.01, 117.73), ("ping pong ball", "speed", 1.4, 0.01, 87.9)]),
    "captured_0027s": ({"red ball": [405, 208, 453, 262]}, ("red ball", 0.07),
                       [("red ball", "speed", 0.2, 0.01, 201.73), ("red ball", "speed", 0.6, 0.01, 66.68),
                        ("red ball", "accel", 0.2, 0.01, 1086.09)]),
    "captured_0034as": ({"left tennis ball": [148, 300, 199, 354], "right tennis ball": [658, 303, 712, 350]},
                        ("left tennis ball", 0.067),
                        [("left tennis ball", "speed", 1.2, 0.01, 82.93), ("right tennis ball", "speed", 1.2, 0.01, 99.47),
                         ("left tennis ball", "speed", 1.3, 0.01, 84.54), ("right tennis ball", "speed", 1.3, 0.01, 92.33)]),
}
QVAL_CAPTIONS = {
    "captured_0021x": "A white ping pong ball on a wooden table. Static camera.",
    "captured_0027s": "A red ball hangs on a string. Static camera.",
    "captured_0034as": "Two tennis balls, a football and a pencil case on a tilting table. Static camera.",
}


def main(root="data"):
    real = Path(root) / "real"
    (Path(root) / "physics_iq").mkdir(parents=True, exist_ok=True)
    real.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (stem, box, cap) in PIQ.items():
        f = Path(root) / "physics_iq" / f"{stem}.mp4"
        if not f.exists():
            urlretrieve(PIQ_URL.format(stem), f)
        manifest.append(dict(id=f"piq_{name}", source="physics-iq", video=str(f), wh=[3840, 2160],
                             input_end_s=PIQ_INPUT_END_S, box=box, caption=cap))
    qdir = Path(root) / "quantiphy_val"
    if not (qdir / "validation_dataset.csv").exists():
        from huggingface_hub import snapshot_download
        snapshot_download("PaulineLi/QuantiPhy-validation", repo_type="dataset", local_dir=qdir)
    for clip, (boxes, prior, _q) in QVAL.items():
        manifest.append(dict(id=f"qval_{clip}", source="quantiphy-val", video=str(qdir / "validation_videos" / f"{clip}.mp4"),
                             wh=[854, 480], start_s=0.0, box=boxes[prior[0]], caption=QVAL_CAPTIONS[clip]))
    (real / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (real / "qval_questions.json").write_text(json.dumps(
        {c: dict(boxes=b, prior=p, questions=q) for c, (b, p, q) in QVAL.items()}, indent=1))
    print(f"{len(manifest)} real clips -> {real / 'manifest.json'}")


if __name__ == "__main__":
    main()
