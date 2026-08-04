# ICT/SMC Objects — Riconoscimento, chiarezza per timeframe, potere predittivo

```
══════════════════════════════════════════════════════════════════════════════
ICT/SMC Objects — Riconoscimento, chiarezza per timeframe, potere predittivo
══════════════════════════════════════════════════════════════════════════════

7 oggetti dalle slide condivise + segnale combinato (sweep->OB->FVG).
Event study: forward return nella direzione implicita, Welch t-test vs
controllo, 4 orizzonti (1/5/10/20 barre), 4 timeframe (1D/4H/1H/15M).

══════════════════════════════════════════════════════════════════════════════
FREQUENZA (eventi totali nel campione, per timeframe)
══════════════════════════════════════════════════════════════════════════════

  Oggetto                       1D        4H        1H       15M
  Swing H/L (sweep)            556     3,211    12,981    54,081
  FVG (BISI/SIBI)              400     2,136     8,215    37,130
  Order Block                1,247     7,408    30,575   123,958
  Breaker Block                790     4,879    20,577    85,612
  Inversion FVG                276     1,543     6,105    28,489
  Power of 3 (AMD)              29       432     1,987     5,122
  COMBO sweep->OB->FVG         411     2,264     9,142    40,705

══════════════════════════════════════════════════════════════════════════════
POTERE PREDITTIVO — orizzonte 1 barre (mean return % nella direzione implicita, * = p<0.05)
══════════════════════════════════════════════════════════════════════════════

  Oggetto                             1D              4H              1H             15M
  Swing H/L (sweep)             -0.098%         -0.018%         -0.004%         -0.002% 
  FVG (BISI/SIBI)               +0.055%         +0.028%         +0.001%         -0.002% 
  Order Block                   -0.013%         -0.015%         -0.003%         -0.002% 
  Breaker Block                 +0.114%         -0.023%         -0.002%         +0.000% 
  Inversion FVG                 +0.124%         -0.038%         -0.010%         -0.008%*
  Power of 3 (AMD)              -0.612%         +0.109%         -0.022%*        -0.005% 
  COMBO sweep->OB->FVG          -0.052%         +0.019%         +0.005%         +0.000% 

══════════════════════════════════════════════════════════════════════════════
POTERE PREDITTIVO — orizzonte 5 barre (mean return % nella direzione implicita, * = p<0.05)
══════════════════════════════════════════════════════════════════════════════

  Oggetto                             1D              4H              1H             15M
  Swing H/L (sweep)             -0.001%         +0.046%         +0.013%         +0.002% 
  FVG (BISI/SIBI)               +0.741%         +0.026%         +0.025%         +0.002% 
  Order Block                   +0.278%         -0.022%         -0.028%*        -0.004% 
  Breaker Block                 +0.436%         +0.013%         +0.003%         -0.001% 
  Inversion FVG                 +1.146%*        +0.029%         -0.031%         -0.010% 
  Power of 3 (AMD)              +2.304%*        +0.462%*        -0.018%         -0.002% 
  COMBO sweep->OB->FVG          +0.596%         -0.114%         +0.019%         +0.002% 

══════════════════════════════════════════════════════════════════════════════
POTERE PREDITTIVO — orizzonte 10 barre (mean return % nella direzione implicita, * = p<0.05)
══════════════════════════════════════════════════════════════════════════════

  Oggetto                             1D              4H              1H             15M
  Swing H/L (sweep)             -0.091%         -0.009%         -0.017%         +0.003% 
  FVG (BISI/SIBI)               +1.292%*        +0.178%         +0.051%*        +0.006% 
  Order Block                   +0.082%         -0.013%         -0.040%*        -0.007%*
  Breaker Block                 +0.604%         +0.062%         +0.012%         -0.006% 
  Inversion FVG                 +0.435%         -0.008%         -0.047%         -0.007% 
  Power of 3 (AMD)              +0.792%         +0.429%*        -0.051%         +0.008% 
  COMBO sweep->OB->FVG          -0.055%         -0.043%         -0.006%         -0.001% 

══════════════════════════════════════════════════════════════════════════════
POTERE PREDITTIVO — orizzonte 20 barre (mean return % nella direzione implicita, * = p<0.05)
══════════════════════════════════════════════════════════════════════════════

  Oggetto                             1D              4H              1H             15M
  Swing H/L (sweep)             -0.570%         -0.105%         -0.047%*        +0.000% 
  FVG (BISI/SIBI)               +2.245%*        +0.193%         +0.094%*        +0.005% 
  Order Block                   +0.028%         -0.081%         -0.015%         -0.013%*
  Breaker Block                 +0.287%         +0.073%         +0.000%         -0.010% 
  Inversion FVG                 +1.030%         -0.096%         +0.012%         -0.011% 
  Power of 3 (AMD)              -2.090%         +0.008%         -0.124%*        -0.002% 
  COMBO sweep->OB->FVG          -0.755%         -0.167%         +0.056%         +0.002% 

══════════════════════════════════════════════════════════════════════════════
[DONE]
══════════════════════════════════════════════════════════════════════════════
```

## Sintesi: dove sembrano "potenti" è dove il campione è troppo piccolo per fidarsene

**Il pattern chiave, visibile in ogni tabella**: gli effetti GRANDI stanno
sul **1D** (campione piccolo, 29-1.247 eventi), gli effetti su **1H/15M**
(campioni enormi, migliaia-centinaia di migliaia di eventi) sono quasi
tutti minuscoli o nulli. Questo è l'opposto di quello che ci si
aspetterebbe da un vero segnale (che dovrebbe reggere, magari attenuato,
anche su campioni grandi) — è invece la firma classica del rumore
statistico su pochi punti dati.

**Prova diretta — incoerenza tra orizzonti sullo stesso oggetto (1D)**:
```
Power of 3 (AMD), 1D, n=29:  1bar -0.61%   5bar +2.30%*   10bar +0.79%   20bar -2.09%
Inversion FVG, 1D, n=276:    1bar +0.12%   5bar +1.15%*   10bar +0.44%   20bar +1.03%
```
Un vero effetto di mercato non dovrebbe CAMBIARE SEGNO passando da 5 a 20
barre di orizzonte sullo stesso identico set di eventi (Power of 3: da
+2.30% a -2.09%). Questo è incompatibile con un edge reale — è la
variabilità di un campione di 29 osservazioni, non un fenomeno
sistematico.

**Su campioni grandi (1H, 15M) dove ci si può fidare della statistica,
gli effetti sono economicamente irrilevanti E a volte di segno SBAGLIATO
rispetto alla teoria ICT**:
```
Order Block, 1H (n=30.575):     5bar -0.028%*   10bar -0.040%*
Power of 3, 1H (n=1.987):       20bar -0.124%*
```
Questi sono STATISTICAMENTE significativi (p<0.05, grazie al campione
enorme) ma NEGATIVI — l'Order Block e il Power of 3 su 1H predicono, in
media, il movimento OPPOSTO a quello atteso dalla teoria (un "OB
rialzista" tende a precedere un ritorno leggermente ribassista, non
rialzista). L'effetto (0.03-0.12%) resta comunque troppo piccolo per
essere tradabile (ben sotto qualunque costo di esecuzione), ma il segno
sbagliato è di per sé una prova che la narrativa "smart money accumula
qui" non trova conferma nei dati.

**L'unico oggetto con un segnale piccolo ma internamente coerente**: la
**FVG su 1H** (n=8.215) è positiva a tutti gli orizzonti testati (5/10/20
barre) e significativa a 10 e 20 barre (+0.051%*, +0.094%*) — l'unico
caso in questo intero studio di un effetto (a) nel verso "giusto" secondo
la teoria, (b) statisticamente significativo, (c) coerente in segno su
più orizzonti, (d) su un campione ampio. Resta comunque minuscolo
(0.05-0.09%, un ordine di grandezza sotto il costo di transazione) —
un'informazione reale ma non sfruttabile da sola.

**Il segnale COMBINATO (sweep -> OB -> FVG in sequenza) non migliora
nulla**: su nessun timeframe supera in modo consistente i singoli
componenti — anzi è spesso più debole e più rumoroso della sola FVG
(es. 1H a 10 barre: combo -0.006% n.s. contro FVG da sola +0.051%*).
Incatenare più oggetti ICT in sequenza NON produce, in questo studio, un
segnale più forte: diluisce il campione (da 8.215 FVG a 9.142 combo — un
numero simile ma un mix diverso di eventi, meno selettivo di quanto la
narrativa "confluenza" suggerirebbe) senza aggiungere potere predittivo.

**Risposta diretta alle tue domande**:
1. **Timeframe più facile per l'identificazione**: dipende dall'oggetto
   (vedi tabella frequenza) — ma la "facilità di riconoscimento" non
   correla con il potere predittivo: gli oggetti più frequenti (Order
   Block, Breaker Block) sono anche quelli col segno sbagliato o nullo.
2. **Potere predittivo**: quasi ovunque assente o economicamente
   irrilevante; l'unico caso credibile è FVG su 1H, minuscolo (~0.05-0.09%).
3. **Segnale combinato**: NON migliora il potere predittivo rispetto ai
   singoli componenti in questo test — anzi lo diluisce leggermente.

Coerente con **tutto** quanto già trovato in questa sessione sulla
famiglia ICT/SMC (Asia-sweep v1-v4, MTF FVG reaction, order-flow
absorption, SMC structure clarity): il riconoscimento algoritmico di
questi pattern è fattibile e anche abbastanza pulito come STRUTTURA (vedi
`smc_structure_clarity.md`), ma la narrativa causale ("smart money
manipola qui, poi distribuisce là") non si traduce in un segnale
predittivo di prezzo utilizzabile, con o senza combinazione in sequenza.
