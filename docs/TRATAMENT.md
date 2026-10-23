# Tratament: contează doza de radioterapie?

Set de date: CFB-GBM (TCIA, CC BY 4.0), 264 pacienți, hărți de doză (RTDOSE).

## 1. Unde crește tumora? (10 pacienți cu imagini)
Zona nouă = voxeli în GTV la momentul progresiei, absenți în GTV la t0. Zona prezisă = top-k voxeli (k = volumul zonei noi).

| Metodă | Medie acoperire |
|---|---|
| Margine uniformă | 58,2 % |
| Model fără doză | 59,1 % |
| Model cu doză (supraviețuire liniar-pătratică, α=0,15, β=0,05) | 59,1 % |

Doza medie în tumora inițială și în zona nouă diferă cu < 0,5 Gy în 9 din 10 cazuri: radioterapia conformală lasă doza aproape plată exact unde se măsoară creșterea, deci un termen de moarte celulară scade u uniform și nu schimbă ordinea voxelilor. Detalii pe pacient: `rezultate_doza_cfb.txt`. Reproducere: `examples/cfb_dose_pilot.py`.

## 2. Cât de repede crește? (129 pacienți, din tabele)
Creștere = ln(V_t1 / V_t0) / săptămâni.

| Doză | n | Vârstă mediană | Volum inițial mediu |
|---|---|---|---|
| 60 Gy | 96 | 65 | 33 cm³ |
| 40,05 Gy | 33 | 75 | 57 cm³ |

Mann–Whitney p = 0,70; regresie ajustată (vârstă, log volum) p = 0,56 pentru doză. Volumul inițial: p = 0,002.

## Limite
- 40,05 Gy se dă vârstnicilor: doza e confundată cu vârsta.
- GTV este volum de planificare, nu recidivă confirmată; cifrele nu se compară direct cu LUMIERE.
- Doar 10 pacienți cu imagini; doar 2 la 40,05 Gy.
- Termenul de doză e un singur model (LQ aplicat o dată); alte forme nu au fost testate.

## Concluzie
În aceste date doza nu aduce informație despre unde sau cât de repede crește tumora. Rezultat negativ.
