# Rezultate: de unde vine fiecare număr din lucrare

Acest folder conține toate rezultatele din Capitolul 4 al lucrării, în fișiere care se pot deschide și verifica.
Tabelele se refac din aceste fișiere cu o singură comandă:

    python scripts/teza/tabele_teza.py        # scrie rezultate/tabele.md

## Ce conține folderul

| Fișier | Ce este |
|---|---|
| `pacienti/ev_Patient-XXX.json` | rezultatul brut pentru fiecare pacient: acoperirea fiecărei metode la 20, 50 și 100 cm³, pentru protocolul A (`"start": "pre"`) și B (`"start"` = săptămâna primului control) |
| `rezumat_pacienti.json` | același lucru, rezumat pe un rând pe pacient și protocol (folosit și de aplicație) |
| `statistici.json` | diferențele dintre metode: medie, interval de încredere 95% (bootstrap), p (Wilcoxon), câți pacienți sunt mai bine / mai rău |
| `timp_de_sosire.json` | acoperirea zonei calculate din timpul de sosire (limita eikonală), pe aceiași pacienți |
| `retele.txt` | ieșirea celor trei exemple pe rețele (financiar, social, economic) |
| `tabele.md` | tabelele din lucrare, refăcute automat din fișierele de mai sus |

## Cheile din fișierele JSON

| Cheie | Metoda |
|---|---|
| `D0.1_V` | modelul Fisher–KPP cu parametrii din lucrare (D alb = 0,1 mm²/zi, ρ = 0,02/zi), zona de V cm³ |
| `marginV` | marginea uniformă de același volum (voxelii cei mai apropiați de tumora de pornire) |
| `staticV` | zona vizibilă pe RMN (T2), fără simulare |
| `m15`, `m15vol` | marginea clinică de 15 mm: acoperirea și volumul ei |
| `eikV` (în `timp_de_sosire.json`) | zona calculată din timpul de sosire (Dijkstra) |
| `Ovol` | volumul recidivei (cm³); recidivele sub 1 cm³ nu intră în medii |
| `D0.03_V`, `D0.3_V`, `consV`, `J1_V`, `J2_V` | teste suplimentare: alte valori ale lui D, zona de consens, termenul cu salturi |

## Corespondența cu lucrarea

| În lucrare | Din ce fișier | Cum |
|---|---|---|
| Tabelul 4.1 (pacienții 062 și 051) | `pacienti/ev_Patient-062.json`, `ev_Patient-051.json` | rândul cu `"start": "pre"` |
| Tabelul 4.2 (cohorta) | `pacienti/*.json` | media pe pacienți, doar cei cu `Ovol ≥ 1` (21 în A, 26 în B) |
| „+11 până la +37 de puncte”, „p < 0,01” | `statistici.json` | cheile `model-stat_V` |
| „echivalent cu marginea în medie” | `statistici.json` | cheile `model-marg_V` (intervalul de încredere conține 0) |
| corelațiile din Secțiunea 4.4 | `timp_de_sosire.json` | corelația pe pacienți între `D0.1_V` și `eikV` |
| Tabelul 4.3 (rețele) | `retele.txt` | `python examples/finance_contagion.py`, `social_cascade.py`, `economic_regions.py` |

## Cum au fost obținute (pașii)

1. **Datele:** setul public LUMIERE (Suter et al., 2022): imagini RMN și segmentările DeepBraTumIA. Datele nu sunt incluse aici (licența cere descărcarea de la sursă).
2. **Pregătirea:** `scripts/teza/export_all.py` aduce imaginile pe grila de 2 mm, calculează substanța albă și cenușie (k-means pe T1) și salvează câte un fișier `.npz` pe pacient.
3. **Simularea și acoperirea:** `scripts/teza/eval_all.py` rulează pentru fiecare pacient modelul (solverul `scripts/teza/fkpp.py`, schema IMEX cu gradient conjugat), construiește zonele de 20, 50 și 100 cm³ și calculează acoperirea pentru fiecare metodă → `pacienti/ev_Patient-XXX.json`.
4. **Statistica:** `scripts/teza/ev_build.py` adună rezultatele și calculează intervalele de încredere (5000 de reeșantionări bootstrap) și testul Wilcoxon → `statistici.json`.
5. **Timpul de sosire:** `scripts/teza/geo_eval.py` calculează zona din limita eikonală, cu același protocol → `timp_de_sosire.json`.
6. **Calibrarea (rezultat negativ):** `scripts/teza/calib.py`.

Scripturile din `scripts/teza/` sunt cele rulate efectiv pentru lucrare; căile spre date din ele (folderul cu LUMIERE) trebuie adaptate pe calculatorul tău.

Aceiași algoritmi există, în formă curată și testată, în pachetul `reactrisk` (`reactrisk/core`, `reactrisk/glioma/evaluate.py`).
