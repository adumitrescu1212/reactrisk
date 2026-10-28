# Cum se reutilizează nucleul pentru alt domeniu

Nucleul cere trei lucruri: un operator de difuzie `A` (matrice rară simetrică cu suma pe linii 0), o stare inițială `u0 ∈ [0,1]` și un buget `k`.

| Glioblastom | Contagiune financiară | Alt exemplu (rețea socială / economie regională) |
|---|---|---|
| voxel | bancă | utilizator / regiune |
| densitate tumorală `u` | grad de dificultate 0..1 | adoptare a unei idei / nivel de șoc economic |
| difuzie între voxeli vecini | pierderi transmise prin expuneri `W_ij` | influență între utilizatori / legături comerciale |
| proliferare `ρ u (1 − u)` | auto-amplificare (vânzări forțate, pierderea încrederii) | efect de rețea cu saturație |
| tumora la diagnostic | băncile lovite primele | sursa inițială |
| zona de `V` cm³ | cele `k` bănci supravegheate | cele `k` noduri pe care le țintești |
| recidiva reală | băncile care cad ulterior | nodurile afectate efectiv |
| marginea uniformă | număr de pași (hop) de la sursă | distanța în rețea |

## Pași
1. Construiește `A`: `graph_operator(W)` pentru o rețea, `grid_operator(D, mask, h)` pentru o grilă.
2. Pune șocul inițial în `u0` (`finance.shock_vector`).
3. Rulează `IMEXSolver(A, rho, dt).run(u0, n_pasi)`.
4. Alege zona cu `top_k(u, k)` și judec-o cu `tie_aware_coverage` față de ce s-a întâmplat.
5. Compară mereu cu o referință simplă la același buget (distanță, expunere directă).

Banc de test comun: `reactrisk.core.benchmark.run(W, cascada, k)`. Adevărul vine dintr-un **alt mecanism** decât modelul (cascadă cu prag, cascadă independentă, model cu prag), deci scorul nu este circular. `ρ` și orizontul se aleg pe prima jumătate a scenariilor și se raportează pe a doua.

## Rezultate (rețele sintetice, 100 de scenarii de test)

| Metodă | Finanțe (`finance_contagion.py`) | Social (`social_cascade.py`) | Economie (`economic_regions.py`) |
|---|---|---|---|
| model reacție–difuzie | **62 %** | **41 %** | **76 %** |
| timp de sosire (eikonal) | 65 % | 40 % | 71 % |
| distanța în rețea (hops) | 45 % | 37 % | 71 % |
| expunere directă | 44 % | 35 % | 68 % |
| gradul nodului | 28 % | 15 % | 10 % |
| doar sursele (șansă) | 10 % | 10 % | 10 % |

Modelul bate distanța în toate trei (p < 1e-6). Rețelele sunt sintetice: arată că metoda se transferă, nu că funcționează pe date reale.

## Limite
Modelul presupune propagare locală și saturare. Dacă fenomenul are salturi mari sau praguri puternice, termenul cu salturi (`core/jump.py`) sau un model de prag sunt mai potrivite.
