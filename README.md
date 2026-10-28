# reactrisk

Reaction–diffusion **risk zones**: given where something is now, rank the places it is most likely to appear next, for a fixed budget.

* **Medical application** – recurrence of glioblastoma from MRI (public LUMIERE dataset).
* **Same algorithm, other domain** – financial contagion in a network of banks.

> Research code. Not a medical device and not financial advice.

## Idea in one minute

A quantity `u` between 0 and 1 (tumour cell density, or distress of a bank) spreads to its neighbours and grows where it is already present:

    du/dt = div(D grad u) + rho u (1 - u)       (Fisher–KPP)

1. Discretise space (voxels or graph nodes) → a sparse operator `A` (`reactrisk.core.operators`).
2. Advance in time with an implicit–explicit scheme, `(I - dt A) u_new = u + dt rho u (1 - u)`, solved by preconditioned CG (`reactrisk.core.solver`). Proven: stable, keeps `u` in [0, 1] if `dt·rho ≤ 1`.
3. Rank the cells by `u`, keep the best `k` (the *zone*), and measure how much of what really happened falls inside (`reactrisk.core.ranking`).

## Install and test

    pip install -e .[dev]
    pytest

## Try it

    python examples/finance_contagion.py          # synthetic bank network, no data needed
    python examples/glioma_synthetic.py           # toy brain, no data needed
    python scripts/prepare_lumiere.py <LUMIERE dir> Patient-062 out/    # needs the public data, see data/README.md
    python examples/glioma_case.py out/Patient-062.npz --start pre --target 50

## What the results say (honestly)

* Glioma, 32 evaluable LUMIERE patients, equal volume: the model is clearly better than the static T2 zone (p < 0.001) and about as good as a uniform margin around the cavity. It is clearly better than the margin only in some cases (e.g. patient 051, recurrence far from the original site). Calibrating D and rho per patient did not help.
* Finance example: on a *synthetic* network the model beats distance and direct-exposure baselines. This shows the method transfers, it is not evidence about real banks.

## Layout

| Folder | Content |
|---|---|
| `reactrisk/core` | operators, IMEX solver, ranking and coverage, jump term (domain independent) |
| `reactrisk/glioma` | MRI labels → model inputs, evaluation protocol, clinical-factor statistics |
| `reactrisk/finance` | bank network, exposure operator, contagion example |
| `docs/ALGORITMI.md` | the algorithms, step by step (Romanian) |
| `docs/GENERALIZARE.md` | how to reuse the core for a new domain (Romanian) |
- [docs/TRATAMENT.md](docs/TRATAMENT.md): test cu doza de radioterapie (CFB-GBM), rezultat negativ

## Pe scurt (RO)

Un model reacție–difuzie estimează unde apare următoarea recidivă (sau următoarea bancă în dificultate). Nucleul (`reactrisk.core`) nu depinde de domeniu; `glioma` și `finance` sunt două aplicații. Detalii în `docs/`.

## License

MIT, see `LICENSE`. LUMIERE data: CC BY 4.0, not redistributed here.

## Aplicația interactivă

`app/index.html` este simulatorul (un singur fișier, rulează în browser, fără instalare), cu parcurs ghidat în 6 pași și „Mod expert”. Deschide-l direct sau activează GitHub Pages (Settings, Pages, branch `main`, folder `/app`) ca să aibă link public.
Conține imagini și segmentări din LUMIERE (Suter et al.); verifică licența și cerința de citare a setului înainte de a face repo-ul public. Nu este un instrument clinic.
