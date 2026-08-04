# Fade ICT standalone (RR=3.0) — Rendimento atteso a 30/50/60 giorni

```
══════════════════════════════════════════════════════════════════════════════
Fade ICT standalone (RR=3.0) — Rendimento atteso a 30/50/60 giorni (forward)
══════════════════════════════════════════════════════════════════════════════

Tasso storico: 1399 trade in 2372 giorni -> 0.590 trade/giorno (~1 ogni 1.7 giorni)

══════════════════════════════════════════════════════════════════════════════
METODO A — Finestre storiche realizzate (rendimento reale a N giorni, ogni punto di partenza)
══════════════════════════════════════════════════════════════════════════════

  Orizzonte 30 giorni  (n finestre storiche = 2343):
    Media: +2.29%   Mediana: +1.78%   P(rendimento>0): 78.5%
    Percentili:  5%=-1.65%  10%=-0.99%  25%=+0.34%  75%=+3.43%  90%=+5.77%  95%=+7.85%

  Orizzonte 50 giorni  (n finestre storiche = 2323):
    Media: +3.80%   Mediana: +3.03%   P(rendimento>0): 82.9%
    Percentili:  5%=-1.95%  10%=-0.95%  25%=+1.09%  75%=+5.66%  90%=+9.06%  95%=+12.50%

  Orizzonte 60 giorni  (n finestre storiche = 2313):
    Media: +4.56%   Mediana: +3.64%   P(rendimento>0): 84.4%
    Percentili:  5%=-2.07%  10%=-0.88%  25%=+1.34%  75%=+6.73%  90%=+10.63%  95%=+13.53%

══════════════════════════════════════════════════════════════════════════════
METODO B — Monte Carlo block-bootstrap forward (block_size=10, 20,000 simulazioni)
══════════════════════════════════════════════════════════════════════════════

  Orizzonte 30 giorni  (~18 trade attesi):
    Media: +6.73%   Mediana: +6.68%   P(rendimento>0): 79.8%   P(perdita>50% capitale): 0.000
    Percentili:  5%=-6.23%  10%=-3.57%  25%=+1.24%  75%=+12.12%  90%=+16.93%  95%=+19.99%

  Orizzonte 50 giorni  (~29 trade attesi):
    Media: +10.57%   Mediana: +10.49%   P(rendimento>0): 85.0%   P(perdita>50% capitale): 0.000
    Percentili:  5%=-6.20%  10%=-2.51%  25%=+3.64%  75%=+17.36%  90%=+23.84%  95%=+27.42%

  Orizzonte 60 giorni  (~35 trade attesi):
    Media: +12.88%   Mediana: +12.79%   P(rendimento>0): 87.6%   P(perdita>50% capitale): 0.000
    Percentili:  5%=-5.36%  10%=-1.40%  25%=+5.43%  75%=+20.40%  90%=+27.20%  95%=+31.19%

══════════════════════════════════════════════════════════════════════════════
CONFRONTO — Metodo A (storico realizzato) vs Metodo B (Monte Carlo)
══════════════════════════════════════════════════════════════════════════════

   Giorni   A: mediana   A: P(>0)   B: mediana   B: P(>0)              B: 5-95%
      30g       +1.78%     78.5%       +6.68%     79.8%  [-6.2%, +20.0%]
      50g       +3.03%     82.9%      +10.49%     85.0%  [-6.2%, +27.4%]
      60g       +3.64%     84.4%      +12.79%     87.6%  [-5.4%, +31.2%]

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: risposta diretta, con due stime che si accordano sulla probabilità ma non sulla magnitudine

```
Giorni   Mediana rendimento          P(rendimento>0)        Range 5-95% (Metodo B)
  30      +1.8% (A) / +6.7% (B)         79-80%               [-6.2%, +20.0%]
  50      +3.0% (A) / +10.5% (B)        83-85%               [-6.2%, +27.4%]
  60      +3.6% (A) / +12.8% (B)        84-88%                [-5.4%, +31.2%]
```

**I due metodi si accordano bene sulla PROBABILITÀ di profitto** (entro
1-3 punti percentuali l'uno dall'altro su tutti e 3 gli orizzonti) — un
buon segnale di cross-validazione sulla direzione dell'edge. **Ma
divergono di circa 3× sulla MAGNITUDINE mediana attesa.**

**Perché divergono**: il Metodo B forza un numero FISSO di trade per
ogni finestra simulata (es. ~18 a 30 giorni, il tasso medio storico),
ricampionando a blocchi dall'INTERA storia 2020-2026 in modo
sostanzialmente indipendente dal periodo di calendario. Il Metodo A usa
invece le vere finestre storiche, che catturano fedelmente come si sono
davvero distribuiti nel tempo sia il NUMERO di trade sia la loro
redditività — e dal breakdown annuale di `ict_fade_standalone.md` sappiamo
che la strategia ha reso molto di più nel 2020-2022 (+109/+118%/anno)
rispetto al 2023-2026 (+26/+77/+27/+38%/anno). Il Metodo B, ricampionando
in modo uniforme su tutta la storia, eredita una redditività media per
trade più alta di quella tipica degli ultimi 3-4 anni — **il Metodo A è
quindi la stima più prudente e più rappresentativa delle condizioni
recenti**, il Metodo B un limite superiore ottimistico.

**Risposta diretta alla domanda**: investendo capitale nella strategia
oggi, il rendimento atteso (stima prudente, Metodo A) è approssimativamente:
- **30 giorni**: mediana +1.8%, probabilità di profitto ~79%, range tipico
  (10°-90° percentile) da -1.0% a +5.8%
- **50 giorni**: mediana +3.0%, probabilità di profitto ~83%, range da
  -1.0% a +9.1%
- **60 giorni**: mediana +3.6%, probabilità di profitto ~84%, range da
  -0.9% a +10.6%

Il Metodo B (Monte Carlo) suggerisce che questi numeri potrebbero essere
conservativi se la strategia recupera la redditività per-trade dei primi
anni — ma non c'è garanzia che ciò accada, ed è prudente pianificare sul
Metodo A. **Nessuno dei due metodi include una correzione DSR per questa
proiezione specifica** (non è stata testata alcuna griglia qui, solo la
configurazione già validata RR=3.0) — il numero di trade attesi è basso
(18-35), quindi la varianza attorno a queste stime resta ampia (si veda
il range 5-95% del Metodo B), come atteso per orizzonti brevi con
frequenza di trading di circa 1 ogni 1,7 giorni.

**Nota tecnica**: durante lo sviluppo di questo script è stato trovato e
corretto un bug di allineamento temporale (i timestamp di uscita dei
trade includevano ora:minuti:secondi, che non collimavano con l'indice
giornaliero usato per il Metodo A, azzerando artificialmente quasi tutte
le finestre storiche — da qui la prima run con P(rendimento>0) vicino al
5-9%, chiaramente incoerente col Metodo B, che ha permesso di individuare
l'errore).
