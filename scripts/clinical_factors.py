"""Clinical factors versus time to first progression (RANO PD) on LUMIERE.

    python scripts/clinical_factors.py <dir with the two LUMIERE csv files>
"""
import sys
import numpy as np
import pandas as pd
from reactrisk.glioma.factors import km_median, logrank_p, cox_fit

d = sys.argv[1].rstrip("/") + "/"
dm = pd.read_csv(d + "LUMIERE-Demographics_Pathology.csv")
er = pd.read_csv(d + "LUMIERE-ExpertRating-v202211.csv")
er.columns = ["Patient", "Date", "lt3", "nm", "rating", "why"]
er["w"] = er.Date.str.extract(r"week-(\d+)").astype(float)
rows = []
for p, g in er.groupby("Patient"):
    g = g.sort_values("w")
    post = g[g.rating == "Post-Op"].why.dropna()
    ext = post.iloc[0].split(",")[0].split(" ")[0] if len(post) else None        # CRET / PRET
    fu = g[(g.w >= 12) & g.rating.isin(["SD", "PD", "PR", "CR"])]
    if not len(fu):
        continue
    pdw = fu[fu.rating == "PD"]
    rows.append(dict(Patient=p, t=pdw.w.iloc[0] if len(pdw) else fu.w.iloc[-1], ev=int(len(pdw) > 0), ext=ext))
r = pd.DataFrame(rows).merge(dm, on="Patient")
r["age"], r["sex"], r["mgmt"] = r["Age at surgery (years)"], r["Sex"], r["MGMT qualitative"]
print(f"{len(r)} patients, {r.ev.sum()} with progression, median {km_median(r.t, r.ev):.0f} weeks")


def compare(name, a, b, la, lb):
    print(f"{name:10s} {la}: n={len(a)} median {km_median(a.t, a.ev):.0f} wk | {lb}: n={len(b)} median {km_median(b.t, b.ev):.0f} wk"
          f" | log-rank p = {logrank_p(a.t, a.ev, b.t, b.ev):.3f}")


compare("MGMT", r[r.mgmt == "methylated"], r[r.mgmt == "not methylated"], "methylated", "not methylated")
compare("Sex", r[r.sex == "female"], r[r.sex == "male"], "female", "male")
compare("Age", r[r.age >= 60], r[r.age < 60], ">=60", "<60")
compare("Resection", r[r.ext == "CRET"], r[r.ext == "PRET"], "complete", "partial")
c = r[r.mgmt.isin(["methylated", "not methylated"]) & r.age.notna() & r.ext.isin(["CRET", "PRET"])]
b, p = cox_fit(np.c_[(c.age - 60) / 10, c.mgmt == "methylated", c.ext == "CRET"], c.t, c.ev)
for name, bi, pi in zip(["age (+10 y)", "MGMT methylated", "complete resection"], b, p):
    print(f"Cox {name:20s} HR = {np.exp(bi):.2f}  p = {pi:.3f}")
