"""Run the approved 12 A100-hour pilot in priority order, inside a time budget.

    python scripts/run_pilot.py --budget-h 11.5        # after setup_runpod.sh (setup time is outside the budget)

Steps run in order; before each one the remaining budget is checked against its estimate (per-clip times are
re-estimated from finished clips as they come in), and a step that does not fit is skipped and logged.
Generation is resumable: finished clips are skipped, so re-running continues where it stopped.
A scorer runs alongside (pilot.score --loop) and data is pushed to HF throughout (setup_runpod.sh started it).
Log: data/logs/pilot.log, budget state: data/logs/budget.json.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
# A100 seconds per clip, from the L40S smoke test / 1.4; replaced by measured medians once clips exist
EST = {"vace13": 284, "cosmos2b": 240, "ltx2b": 27, "cosmos14b": 700, "p7_ltx2b": 6, "p7_wan22_5b": 45}
LOAD_S = 150                                       # model load per step


def gen(model, exp, limit, seeds):
    n = (limit or 7) * len(seeds)
    return dict(name=f"{exp}:{model}", kind="gen", model=model, clips=n, out=[f"runs/{exp}/{model}"],
                cmd=[PY, "-m", "pilot.generate", "--model", model, "--exp", exp, "--seeds", *map(str, seeds)]
                + (["--limit", str(limit)] if limit else []))


def cmd(name, args, est_s):
    return dict(name=name, kind="cmd", est_s=est_s, cmd=[PY, *args])


PLAN = [  # priority order: the gates that matter most come first (approved plan, 2026-10-17)
    gen("vace13", "b1", 20, [0]), gen("cosmos2b", "b1", 20, [0]), gen("ltx2b", "b1", 30, [0, 1]),
    gen("vace13", "b5", 15, [0]), gen("cosmos2b", "b5", 15, [0]), gen("ltx2b", "b5", 30, [0, 1]),
    gen("vace13", "b1real", None, [0]), gen("cosmos2b", "b1real", None, [0]), gen("ltx2b", "b1real", None, [0]),
    cmd("probe-hspeed:ltx2b", ["-m", "pilot.probe", "--model", "ltx2b", "--exps", "b1", "--target", "hspeed"], 300),
    cmd("probe-hspeed:cosmos2b", ["-m", "pilot.probe", "--model", "cosmos2b", "--exps", "b1", "--target", "hspeed"], 300),
    dict(name="steer:ltx2b", kind="gen", model="ltx2b", clips=40, out=["runs/steer/ltx2b"],
         cmd=[PY, "-m", "pilot.steer", "--model", "ltx2b", "--scenes", "10", "--alphas", "-2", "-1", "1", "2"]),
    dict(name="steer:cosmos2b", kind="gen", model="cosmos2b", clips=6, out=["runs/steer/cosmos2b"],
         cmd=[PY, "-m", "pilot.steer", "--model", "cosmos2b", "--scenes", "3", "--alphas", "-2", "2"]),
    dict(name="p7:ltx2b", kind="gen", model="p7_ltx2b", clips=30, out=["runs/p7/ltx2b_noise", "runs/p7/ltx2b_inv"],
         cmd=[PY, "-m", "pilot.invert", "--model", "ltx2b"]),
    dict(name="p7:wan22_5b", kind="gen", model="p7_wan22_5b", clips=30, out=["runs/p7/wan22_5b_noise", "runs/p7/wan22_5b_inv"],
         cmd=[PY, "-m", "pilot.invert", "--model", "wan22_5b"]),
    # reordered 2026-10-18 (user): timing and P8 before B2-nl / B-ratio, which are trimmed to fit the budget
    cmd("p11:timing", ["-c", "import subprocess,sys; sys.exit(subprocess.call(['bash','scripts/time_extra.sh']))"], 1800),
    cmd("p8:cosmos2b", ["-m", "pilot.likelihood", "--scenes", "5"], 900),
    gen("vace13", "b2nl", 10, [0]), gen("cosmos2b", "b2nl", 5, [0]), gen("ltx2b", "b2nl", 30, [0]),
    gen("vace13", "bratio", 3, [0]), gen("cosmos2b", "bratio", 3, [0]), gen("ltx2b", "bratio", 10, [0]),
]
FINAL_PROBES = [(m, e, t) for m in ("vace13", "cosmos2b", "ltx2b")
                for e, t in ((["b1"], "hspeed"), (["b1", "b5"], "g"), (["b1", "b5"], "e"))] + \
               [(f"{m}_{k}", ["p7"], "hspeed") for m in ("ltx2b", "wan22_5b") for k in ("noise", "inv")]


def measured(model):
    """Median wall seconds per clip of finished runs for this model (None if none yet)."""
    walls = []
    for f in Path("runs").glob(f"*/{model.removeprefix('p7_')}*/*/meta.json"):
        w = json.loads(f.read_text(encoding="utf-8")).get("wall_s")
        if w and (("p7_" in model) == ("/p7/" in f.as_posix())):
            walls.append(w)
    return statistics.median(walls) if len(walls) >= 3 else None


def estimate(step):
    if step["kind"] == "cmd":
        return step["est_s"]
    per = measured(step["model"]) or EST[step["model"]]
    return LOAD_S + step["clips"] * per


def log(msg, state=None):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open("data/logs/pilot.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if state is not None:
        Path("data/logs/budget.json").write_text(json.dumps(state, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-h", type=float, default=11.5)
    ap.add_argument("--reserve-h", type=float, default=0.4, help="kept for final scoring, probes and push")
    a = ap.parse_args()
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    start_file = Path("data/logs/pilot_start")
    if not start_file.exists():
        start_file.write_text(str(time.time()))
    start = float(start_file.read_text())
    done_flag = Path("data/logs/pilot_gen_done")
    done_flag.unlink(missing_ok=True)
    scorer = subprocess.Popen([PY, "-m", "pilot.score", "--loop", "300", "--until-file", str(done_flag)],
                              stdout=open("data/logs/score.log", "a"), stderr=subprocess.STDOUT)
    state = dict(budget_h=a.budget_h, steps={})
    for step in PLAN:
        if step.get("out") and all(len(list(Path(o).glob("*/meta.json"))) >= step["clips"] for o in step["out"]):
            state["steps"][step["name"]] = dict(status="done earlier")
            log(f"DONE-EARLIER {step['name']}", state)
            continue
        used = (time.time() - start) / 3600
        est = estimate(step) / 3600
        left = a.budget_h - a.reserve_h - used
        if est > left:
            state["steps"][step["name"]] = dict(status="skipped (budget)", est_h=round(est, 2), left_h=round(left, 2))
            log(f"SKIP {step['name']}: needs ~{est:.2f} h, {left:.2f} h left", state)
            continue
        log(f"START {step['name']} (est {est:.2f} h, used {used:.2f} h)")
        t0 = time.time()
        with open(f"data/logs/step_{step['name'].replace(':', '_')}.log", "a") as out:
            try:
                rc = subprocess.run(step["cmd"], stdout=out, stderr=subprocess.STDOUT,
                                    timeout=max(est * 3600 * 2.5, 900)).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        dt = (time.time() - t0) / 3600
        state["steps"][step["name"]] = dict(status="ok" if rc == 0 else f"failed ({rc})", hours=round(dt, 2),
                                            est_h=round(est, 2))
        state["used_h"] = round((time.time() - start) / 3600, 2)
        log(f"END {step['name']}: rc={rc}, {dt:.2f} h (used {state['used_h']} h)", state)
    done_flag.write_text("done")
    scorer.wait()
    for model, exps, target in FINAL_PROBES:
        r = subprocess.run([PY, "-m", "pilot.probe", "--model", model, "--exps", *exps, "--target", target],
                           capture_output=True, text=True)
        log(f"probe {model} {target}: " + (r.stdout.strip().splitlines() or ["(no output)"])[-1][:200])
    subprocess.run([PY, "scripts/sync.py", "push"])
    state["used_h"] = round((time.time() - start) / 3600, 2)
    log(f"PILOT DONE, {state['used_h']} h used", state)


if __name__ == "__main__":
    main()
