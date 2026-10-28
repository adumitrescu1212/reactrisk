"""Reface tabelele din Capitolul 4 al lucrarii, doar din fisierele din rezultate/ (nu are nevoie de imagini).

    python scripts/teza/tabele_teza.py          ->  scrie rezultate/tabele.md
"""
import json, glob, os, re
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'rezultate')
V = (20, 50, 100)
pc = lambda x: f"{round(100 * x)}%"


def rows():
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'pacienti', 'ev_Patient-*.json'))):
        p = re.search(r'Patient-(\d+)', f).group(1)
        for x in json.load(open(f)):
            q = x['Q']
            out.append(dict(p=p, prot='A' if x['start'] == 'pre' else 'B', start=x['start'], target=x['target'], O=x['Ovol'],
                            model=[q[f'D0.1_{v}'] for v in V], marg=[q[f'margin{v}'] for v in V], stat=[q[f'static{v}'] for v in V],
                            m15=x['m15'], m15vol=x['m15vol']))
    return out


R = rows()
L = ["# Tabelele din Capitolul 4, refacute din rezultate/\n"]

L.append("## Tabelul 4.1 - studii de caz (protocolul A, din RMN-ul preoperator)\n")
L.append("| Pacient | Metoda | 20 cm3 | 50 cm3 | 100 cm3 |\n|---|---|---|---|---|")
for p in ('062', '051'):
    r = [x for x in R if x['p'] == p and x['prot'] == 'A'][0]
    for k, n in (('model', 'Modelul Fisher-KPP'), ('marg', 'Margine uniforma'), ('stat', 'Zona vizibila (T2)')):
        L.append(f"| {p} (tinta: sapt. {r['target']}) | {n} | " + " | ".join(pc(v) for v in r[k]) + " |")
    L.append(f"| {p} | Margine clinica 15 mm | {pc(r['m15'])} cu {round(r['m15vol'])} cm3 tratati | | |")

L.append("\n## Tabelul 4.2 - acoperirea medie pe cohorta (recidive >= 1 cm3)\n")
L.append("| Protocol | n | Metoda | 20 cm3 | 50 cm3 | 100 cm3 |\n|---|---|---|---|---|---|")
for prot in ('A', 'B'):
    G = [x for x in R if x['prot'] == prot and x['O'] >= 1]
    for k, n in (('model', 'Modelul Fisher-KPP'), ('marg', 'Margine uniforma'), ('stat', 'Zona vizibila (T2)')):
        m = np.mean([x[k] for x in G], axis=0)
        L.append(f"| {prot} | {len(G)} | {n} | " + " | ".join(pc(v) for v in m) + " |")

S = json.load(open(os.path.join(ROOT, 'statistici.json')))
L.append("\n## Testele statistice (diferenta medie, IC 95% bootstrap, p Wilcoxon, pacienti mai bine / mai rau)\n")
L.append("| Protocol | Comparatie | Volum | Diferenta (puncte) | IC 95% | p | + / - |\n|---|---|---|---|---|---|---|")
for prot, key in (('A', 'pre'), ('B', 'ctrl')):
    for comp, n in (('model-stat', 'model - zona T2'), ('model-marg', 'model - margine')):
        for v in V:
            d, lo, hi, p, up, dn = S[key][f'{comp}_{v}']
            L.append(f"| {prot} | {n} | {v} | {100*d:+.1f} | [{100*lo:+.1f}; {100*hi:+.1f}] | {p:.3f} | {up} / {dn} |")

G = json.load(open(os.path.join(ROOT, 'timp_de_sosire.json')))
L.append("\n## Interpretare (Sectiunea 4.4): corelatia pe pacienti cu modelul\n")
L.append("| Protocol | Volum | model vs timp de sosire | model vs margine |\n|---|---|---|---|")
for prot in ('A', 'B'):
    g = [v for k, v in G.items() if (k.endswith('|pre')) == (prot == 'A') and v['Ovol'] >= 1]
    for v in V:
        m = [x[f'D0.1_{v}'] for x in g]
        L.append(f"| {prot} | {v} | {np.corrcoef(m, [x[f'eik{v}'] for x in g])[0,1]:.2f} | {np.corrcoef(m, [x[f'margin{v}'] for x in g])[0,1]:.2f} |")

open(os.path.join(ROOT, 'tabele.md'), 'w').write("\n".join(L) + "\n")
print("\n".join(L))
