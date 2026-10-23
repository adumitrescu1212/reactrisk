# Algoritmii, pe scurt

Notații: `u` = densitatea (0..1), `D` = cât de repede se împrăștie, `ρ` = cât de repede crește, `h` = pasul grilei, `Δt` = pasul de timp.

## 1. Operatorul de difuzie (`core/operators.py`)
Pentru fiecare pereche de vecini `i, j` din domeniu: `w_ij = 2 D_i D_j / (D_i + D_j) / h²`.
`(A u)_i = Σ_j w_ij (u_j − u_i)`. Pe graf, `w_ij` sunt direct expunerile. Proprietăți (verificate în `tests/`): `A` simetrică, suma pe fiecare linie este 0.

## 2. Pasul IMEX (`core/solver.py`)
`(I − Δt A) u^{n+1} = u^n + Δt ρ u^n (1 − u^n)`.
Sistemul se rezolvă cu gradient conjugat precondiționat (Jacobi). `M = I − ΔtA` este simetrică pozitiv definită; `κ(M) ≤ 1 + 12 Δt D_max / h²` (3D). Dacă `Δt ρ ≤ 1` și `u^0 ∈ [0,1]`, atunci `u^n ∈ [0,1]`.

## 3. Starea inițială și domeniul (`glioma/preprocess.py`)
`u0 = 0,8` în nucleu (necroză + captare), `0,16` în edem, 0 în rest. `D = D_alb` în substanța albă, `D_alb/10` în cea cenușie (clase din k-means pe T1). Domeniul = țesut cerebral fără cavitatea de rezecție (nucleul dilatat cu 1 voxel).

## 4. Zona de risc (`core/ranking.py`)
Se ordonează voxelii după `u` și se păstrează primii `V / 8 mm³` (20, 50 sau 100 cm³). La scoruri egale se împarte proporțional (`tie_aware_coverage`).

## 5. Acoperirea
`acoperire = |zonă ∩ recidivă| / |recidivă|`, la același volum pentru toate metodele. Referințe: marginea uniformă (distanță față de nucleu), zona T2 statică (`u0`), margine clinică de 15 mm.

## 6. Termenul cu salturi (`core/jump.py`, opțional)
`u ← (1 − μ) u + μ K u`, cu `K` stocastică pe linii (neteziere gaussiană normalizată). Păstrează `u ∈ [0,1]` dacă `μ ≤ 1`.

## 7. Consens (`core/ranking.consensus`)
Un voxel intră în zonă dacă este în topul modelului pentru cel puțin 2 din 3 valori ale lui `D` (`D/3`, `D`, `3D`).

## Evaluare (`glioma/evaluate.py`)
Protocol A: pornire din tumora preoperatorie, țintă prima progresie (RANO). Protocol B: pornire din primul control. Se exclud recidivele < 1 cm³. Intervale de încredere bootstrap (5000) și test Wilcoxon pereche.

## Factori clinici (`glioma/factors.py`)
Mediana Kaplan–Meier a timpului până la prima progresie, test log-rank și model Cox. Pe 77 de pacienți LUMIERE, MGMT metilat se asociază cu progresie mai târzie (risc relativ 0,50, p = 0,017); vârsta, sexul și extinderea rezecției nu arată o legătură clară.

## Viteza frontului
Asimptotic `2√(Dρ)`. La timp finit frontul rămâne în urmă: `x(t) = 2√(Dρ)t − 1,5√(D/ρ) ln t` (corecția Bramson), verificat numeric în `tests/test_core.py`.
