"""Run the evaluation on one prepared case (an .npz made by scripts/prepare_lumiere.py).

    python examples/glioma_case.py Patient-062.npz --start pre --target 50
"""
import argparse
import numpy as np
from reactrisk.glioma import run_case

ap = argparse.ArgumentParser()
ap.add_argument("npz"); ap.add_argument("--start", default="pre", help="'pre' or a week number")
ap.add_argument("--target", type=int, required=True, help="week of the first progression (PD)")
a = ap.parse_args()
z = np.load(a.npz)
obs = {int(k[4:]): z[k] for k in z.files if k.startswith("obs_")}
start = a.start if a.start == "pre" else int(a.start)
r = run_case(z["tissue"], z["pre"], z["post"], obs, start, a.target)
if r is None:
    raise SystemExit("recurrence is empty inside the domain")
print(f"recurrence {r['recurrence_cm3']:.1f} cm3, start={r['start']}, target week {r['target']}")
print(f"{'method':14s}" + "".join(f"{v:>9d} cm3" for v in (20, 50, 100)))
for name in ("static", "margin", "D0.1", "consensus"):
    print(f"{name:14s}" + "".join(f"{r['coverage'][f'{name}_{v}']:>13.1%}" for v in (20, 50, 100)))
print(f"15 mm clinical margin: {r['margin15mm']:.1%} of the recurrence, but {r['margin15mm_cm3']:.0f} cm3")
