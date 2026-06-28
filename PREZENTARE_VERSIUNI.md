# Evoluția Trackerului — Jurnal de Lucru

**Alexandru Galea**  
**Competiție:** VOT2026 (Visual Object Tracking Challenge)  
**Hardware:** RTX 3080 10 GB, WSL2

---

## Ce încerc să fac și de ce

VOT2026 este o competiție internațională de urmărire a obiectelor în video. Se dă o secvență video și masca obiectului pe primul frame; trackerul trebuie să întoarcă masca corectă pe fiecare frame următor. Ce complică lucrurile față de urmărirea clasică este că VOTS (varianta cu ocluzie și dispariție) cere ca trackerul să știe și **când obiectul nu mai e în cadru** — să întoarcă mască goală în loc să halucineze. Asta adaugă un nivel complet nou de dificultate față de ce știam eu din cursuri.

Am ales să pornesc de la **DAM4SAM**, câștigătorul VOTS2024. Mi s-a părut mai util să înțeleg și să îmbunătățesc un sistem existent și validat decât să construiesc ceva de la zero fără experiență cu domeniul. DAM4SAM folosește **SAM2** (Segment Anything Model 2 de la Meta) ca backbone — un model de segmentare video pre-antrenat pe sute de milioane de imagini. Ideea DAM4SAM e să adauge deasupra un mecanism de memorie „distractor-aware": când trackerul e suficient de sigur că a găsit obiectul curat, adaugă acel frame în memoria de condiționare (DRM), care ajută re-detecția ulterioară.

---

## Cum testez — metodologia și metricile

### Setul de evaluare rapidă: hard-22

Nu am putut testa fiecare modificare pe toate cele 170 de secvențe din dataset — o rulare completă ia 10–24 de ore pe GPU. Așa că am construit un **set de 22 de secvențe dificile** (hard-22) pe care le-am ales manual să acopere principalele tipuri de eșec:

- secvențe cu **absențe lungi** (obiectul dispare și revine după sute de frame-uri): rabbit, motorcycle-18, seagull
- secvențe cu **distractori vizuali** similari cu obiectul: fish2, colorfish, bull
- secvențe cu **mișcare rapidă și imprevizibilă**: uav1, racing-16, tank-16
- secvențe cu **schimbare de aspect** (iluminare, deformare): wubble_bubble, crab-6, babychimp

O rulare pe hard-22 durează ~2 ore, față de 20+ pentru toate 170. E suficient să văd rapid dacă o idee ajută sau strică.

### Metrica: ce înseamnă dR

Metrica oficială a competiției este **Q** — un scor de calitate compozit calculat de `vot evaluate` care combină acuratețea segmentării pe frame-urile cu obiect cu penalizarea pentru frame-urile cu obiect prezent unde trackerul a dat mască goală (false absențe) și invers. Q=1 ar fi perfect, Q=0.223 a fost scorul nostru pe setul de validare cu GT public.

Pentru experimentele intermediare am folosit **dR** — diferența în rata de frame-uri goale față de baseline. Concret:

- se compară trackerul meu cu baseline-ul (DAM4SAM original) pe aceleași secvențe
- `dR` negativ = trackerul meu dă mai puține frame-uri goale decât baseline → mai bun
- `dR` pozitiv = trackerul meu dă mai multe frame-uri goale → mai rău

De exemplu, rabbit cu `dR = +0.36` înseamnă că al meu a ratat cu 36% mai multe frame-uri decât baseline pe secvența respectivă. E o metrică proxy, nu identică cu Q, dar corelată și calculabilă rapid fără GT complet.

### Cum rulez efectiv

Fiecare versiune are wrapper-ul ei VOT (`vot_wrapper_v3.py` etc.) și e înregistrată în `trackers.ini`. Rulez cu:

```bash
python run_eval.py --tracker v3 --sequences hard_22.txt --out results/v3_hard22
```

Scripul pornește trackerul, îl trece prin fiecare secvență, colectează măștile de output și calculează dR per secvență față de rezultatele baseline deja stocate. La final văd un tabel cu câștig/pierdere per secvență și media.

---

## Modelul de bază: ce am moștenit și ce am reparat înainte de orice experiment

Înainte de a putea face orice experiment semnificativ, am petrecut câteva zile rezolvând probleme de infrastructură pe care nu le anticipasem.

**Problema cu memoria GPU pe secvențe lungi.** Pe hat-5 (8.000 de frame-uri) și robot-19 (5.797 de frame-uri), trackerul se oprea cu OOM (out-of-memory) spre final. Cauza: DRM-ul nu avea nicio limită — după mii de frame-uri se acumulau sute de tensori de features în GPU. Am pus un cap de 30 de frame-uri DRM cu evicție FIFO (cel mai vechi pleacă când se umple). Frame 0 (inițializarea) se păstrează mereu.

**Problema cu `torch.cuda.empty_cache()`.** Codul original chema asta la fiecare frame. Nu știam inițial ce face exact, dar am calculat: la ~75ms/apel pe 288.000 de frame-uri = 6 ore pierdute pe rularea de 170 de secvențe. E un apel care forțează GPU-ul să returneze memoria nefolosită la OS — complet inutil când pattern-ul de memorie e stabil de la un frame la altul. L-am eliminat.

**Problema cu acumularea de outputs.** SAM2 reținea toate output-urile non-condiționate în RAM/GPU de la frame 0. Am adăugat o fereastră glisantă care șterge ce e mai vechi de N frame-uri — păstrăm suficient pentru re-detecție, eliminăm restul.

**Bug-ul cu protocolul TRAX** (specific la portul pe SAM3). VOT toolkit lansează trackerul ca subprocess și comunică prin stdout. SAM3 folosea `tqdm` pentru un progress bar care scria caractere `\r` pe stderr. Toolkit-ul redirecționa stderr pe stdout. Rezultat: caractere de control în mijlocul mesajelor de protocol → parser-ul TRAX vedea date corupte și oprea trackerul după frame 2. Fix: `tqdm_disable=True`. Asta m-a costat o zi întreagă de debugging cu file-logging manual ca să înțeleg ce se întâmplă.

---

## V2 — Prima Idee: Motion-Aware Selection

**Fișier:** [dam4sam/dam4sam_tracker_v2.py](dam4sam/dam4sam_tracker_v2.py)

### De unde a venit ideea

Am citit SAMURAI (ICLR 2025) care propune să folosești un filtru Kalman pentru a re-ordona măștile candidate SAM2 după consistența de mișcare. Ideea mi s-a părut intuitivă: dacă SAM2 e confuz între două măști (obiect real vs. distractor vizual similar), predicția de mișcare ar trebui să discrimineze — distractor-ul stă pe loc, obiectul se mișcă predictibil.

### Ce am implementat

SAM2 generează mai multe măști candidate la fiecare frame. DAM4SAM original le sortează pur după `sam_iou` (IoU-ul prezis de SAM2). Am adăugat un filtru Kalman inițializat din bbox-ul măștii de pe frame 0, care la fiecare frame prezice bbox-ul așteptat. Scorul de selecție devine:

```
scor = 0.80 × sam_iou + 0.20 × motion_iou
```

Am adăugat și `keep_largest_component()` pe masca de output — elimina blob-urile mici detașate care apareau uneori lângă obiect.

KF-ul se actualiza la fiecare frame cu bbox-ul ales, fără nicio verificare calitativă.

### Rezultate

Am rulat pe hard-22 și am văzut ceva ce mă așteptam parțial, dar nu în toată amploarea:

| Secvență | dR față de base | Ce s-a întâmplat |
|----------|-----------------|-----------------|
| seagull | −0.33 | ✅ KF ajuta — pescăruș în zbor cu mișcare rapidă |
| fish2 | −0.20 | ✅ KF ajuta — mai mulți pești asemănători |
| tank-16 | −0.11 | ✅ |
| rabbit | **+0.36** | ❌ tracker pierdut complet |
| bull | **+0.30** | ❌ |
| uav1 | **+0.16** | ❌ |
| **Total net** | **~0** | anulat complet |

Net zero. Câștigurile și pierderile se anulau aproape perfect.

### Ce am înțeles

Pierderile aveau un pattern clar: pe secvențele unde SAM2 era deja sigur (gap mare între candidata 1 și candidata 2), blend-ul de 0.20 motion intervenea oricum și uneori alegea greșit. O singură alegere greșită otrăvea viteza KF-ului cu o direcție falsă — iar KF-ul se „bloca" pe distractor timp de zeci de frame-uri înainte să se recupereze sau nu.

Problema nu era că KF nu ajută. Problema era că îl consultam **tot timpul**, chiar și când n-aveam nevoie. Soluția evidentă în retrospectivă: folosim KF **numai când SAM2 însuși e nesigur**.

---

## V3 — Reparații Chirurgicale ✅ (candidat final de submisie)

**Fișier:** [dam4sam/dam4sam_tracker_v3.py](dam4sam/dam4sam_tracker_v3.py)

V3 nu pornește de la V2 ci direct de la baza. Am scris-o de la zero cu lecțiile din V2. Sunt trei schimbări față de DAM4SAM original.

### Schimbarea 1: KF ca tiebreaker, nu blend global

Comportamentul default rămâne identic cu baseline: `argmax(sam_iou)`. Kalman intră în calcul **numai** dacă diferența dintre primele două IoU-uri e mai mică de `TIE_MARGIN=0.15` — adică SAM2 însuși nu e convins. Și numai dacă KF-ul e „sănătos" (nu a pierdut obiectul recent).

Consecința importantă: pe frame-urile unde SAM2 e sigur, V3 se comportă exact ca baseline-ul. Nu poate pierde mai mult decât baseline-ul pe secvențele ușoare.

### Schimbarea 2: Ciclul de viață al KF reparat

V2 actualiza KF la fiecare frame indiferent de calitatea detecției. V3 actualizează KF **numai** dacă masca curentă trece trei verificări:

- IoU prezis peste 0.70 — detecție de încredere
- cel puțin 10 pixeli — obiect vizibil
- dimensiunea în intervalul `[0.5×, 2×]` față de mediana recentă — fără salturi bruște de dimensiune

Dacă trackerul pierde obiectul 5 frame-uri la rând, KF-ul e marcat ca nesigur și reinițializat la prima detecție bună în loc să continue să „coasting-uiască" cu viteza veche.

### Schimbarea 3: Post-procesare condiționată

`keep_largest_component` se aplica necondiționat în V2. Problema: un pescăruș în zbor sau un iepure parțial ocluzat de gard are legitim două componente în mască. Cu keep_largest tăiam jumătate din obiect. V3 aplică filtrarea **numai** dacă componenta a doua e mai mică de 30% din prima — adică e clar un blob spurios, nu o parte legitimă a obiectului.

### Rezultate

Câștigurile din V2 se păstrează (seagull, fish2), pierderile dispar (rabbit, bull revin la nivelul baseline-ului). **V3 devine candidatul de referință.**

Am ales să moștenesc direct din baza, nu din V2, pentru că V3 rescrie complet logica de selecție și nu mai are sens să porteze codul V2.

### Ce a apărut ulterior: V3b

Rulând V3 pe mai multe secvențe, pe **babychimp** a apărut o regresie de +0.409 — catastrofă. Am inspectat frame cu frame și am văzut: la frame 622, gap-ul top-2 IoU era 0.08 — sub `TIE_MARGIN=0.15`, deci KF a intrat în decizie. A ales greșit. KF-a „prins" o altă maimuță din cadru. Derivare completă în restul secvenței.

**V3b** schimbă un singur parametru: `TIE_MARGIN = 0.05`. Logica: pe secvențele grele (fish2, racing-16) SAM2 e confuz și gap-uri de 0.05 apar des — KF tot intervine și ajută. Pe babychimp, SAM2 e de obicei sigur și gap-ul e rar sub 0.05 — KF nu mai intră → fără derivare. 18 linii de cod, o clasă care moștenește V3 și schimbă o constantă.

---

## V4 — Automat de Stare Prezență / Absență

**Fișier:** [dam4sam/dam4sam_tracker_v4.py](dam4sam/dam4sam_tracker_v4.py)

### Problema pe care V3 nu o rezolvă

V3 e bun la a alege masca corectă dintre candidatele SAM2. Dar nu face nimic special când **obiectul iese din cadru**. Iar VOTS2026 testează exact asta.

Ce se întâmplă în baseline când obiectul dispare: SAM2 halucinează o mască pe cel mai apropiat distractor și „crede" că e obiectul. Această mască falsă poate fi adăugată în DRM (dacă trece pragurile de calitate) — poluând memoria. Când obiectul revine, trackerul s-a „obișnuit" cu distractor-ul și nu mai recunoaște obiectul real.

Soluția pe care am gândit-o: un automat de stare cu două stări, PRESENT și ABSENT, cu histerezis.

### Ce am implementat

Automatul funcționează pe baza a trei semnale pe care SAM2 le produce oricum, gratis:

**`object_score_logits`** — SAM2 are un cap de rețea antrenat să detecteze ocluzii. Dacă logit-ul e negativ, SAM2 însuși crede că obiectul nu e vizibil. Acesta e semnalul pe care mă bazez cel mai mult.

**`m_iou`** — IoU-ul prezis al măștii alese. Dacă e sub 0.50, trackerul e nesigur de propria detecție.

**`appearance_sim`** — corelație între histograma HSV a regiunii mascate curente și template-ul de la inițializare. Filtrează distractorii vizuali care arată diferit față de obiectul original.

```
PRESENT → dacă 3 frame-uri consecutive sunt "rele" → ABSENT  (mască goală)
ABSENT  → dacă 2 frame-uri consecutive sunt "bune" → PRESENT (reinit KF, reia)
```

În starea ABSENT nu adăugăm nimic în DRM — igiena memoriei.

### Rezultate — și ce nu a mers

Net dR V4 față de V3 = **+0.050** pe 13/22 secvențe. Mai rău decât V3.

Două exemple concrete care arată bug-ul:

- **bull**: V3=0.10, V4=0.32. Taurul e mic și se mișcă mult — histograma HSV a regiunii mascate variază rapid. `APP_SIM_THR=0.45` triggerea fals ABSENT deși taurul era perfect vizibil în cadru.
- **wubble_bubble**: V3=0.28, V4=0.41. Obiect care-și schimbă forma și aspectul pe parcurs — template-ul de la inițializare devine tot mai depărtat de aspectul curent, deci `app_sim` cade sub prag.

Problema era că foloseam `APP_SIM` și `m_iou` **la declararea absenței** — prea mulți factori zgomotoși, prea multe false pozitive. Pe secvențele cu absențe reale lungi (motorcycle-18, seagull) V4 câștigă față de V3. Dar pe secvențele fără absențe (bull, wubble_bubble) produce regresii.

Concluzie: mecanismul de absență e util, dar **criteriul de declarare e greșit**. Trebuie simplificat.

---

## V5 — Memorie Adaptivă

**Fișier:** [dam4sam/dam4sam_tracker_v5.py](dam4sam/dam4sam_tracker_v5.py)

V5 rezolvă o altă problemă descoperită pe parcurs, independent de bug-ul din V4.

### Problema ferestrei fixe de memorie

SAM2 reține ultimele N frame-uri procesate ca „memorie non-condiționată" — le consultă pentru consistență temporală. Dacă N e prea mic, uită aspectul stale al distractorilor → câștigă fish2 și seagull. Dacă N e mare, păstrează ancorele din înainte de absență → câștigă rabbit la re-detecție. Nu poți câștiga ambele cu o singură valoare fixă.

**Ipoteza**: dacă legăm N de starea automatului de absență, putem câștiga ambele:

```
PRESENT → fereastră scurtă (30 cadre)  — uită aspectul stale
ABSENT  → fereastră lungă (400 cadre)  — păstrează ancorele pre-absență
```

Logica: la debutul absenței, frame-urile cu obiect prezent sunt încă în fereastra scurtă. Comutarea la fereastra lungă le îngheață în loc să le ștergă.

### Ce s-a întâmplat în practică

A apărut un efect neprevăzut. `SHORT=30` ștergea ancorele pre-absență **înainte** de comutare — când obiectul începea să se degradeze dar încă nu declaraserăm ABSENT. Rabbit cu V5/V6 = 1.157, mai rău decât **baseline-ul** 1.013.

Am testat și `SHORT=300` (identic cu V3) cu `LONG=400`: rabbit se repara, dar motorcycle-18 pierdea avantajul. Prin eliminare, câștigul pe motorcycle-18 nu vine din fereastra scurtă — vine din **detecția absenței** în sine. Fereastra scurtă e de fapt dăunătoare.

Aceasta a clarificat că V5 moștenea și bug-ul din V4 de declarare falsă, pe lângă problema ferestrei. Cele două efecte se suprapuneau și era greu de diagnosticat separat.

---

## V6 — Absență Conservatoare

**Fișier:** [dam4sam/dam4sam_tracker_v6.py](dam4sam/dam4sam_tracker_v6.py)

### Ce încearcă să rezolve

V6 pleacă de la diagnosticul V4: criteriul de declarare a absenței e prea agresiv. Fix: declarăm ABSENT **numai pe `obj_logit` de la SAM2** — singurul semnal antrenat explicit pentru ocluzie, mai puțin zgomotos decât histograma HSV sau IoU-ul prezis.

Trei schimbări față de V4:

**Declarare conservatoare:**
```python
def _is_good_declaration(self, n_pix, obj_logit, m_iou, app_sim):
    return n_pix >= self.MIN_PIX and obj_logit > 0
    # fără ABS_IOU_THR, fără APP_SIM_THR
```

Plus `ABSENT_K = 5` (față de 3 în V4) — trebuie 5 frame-uri consecutive cu `obj_logit < 0` să declar ABSENT, nu 3.

**Re-detecție permisivă cu aspect:** la revenire, verific aspectul cu un prag mai mic (0.30 față de 0.45) — vreau să nu ratez re-apariții reale, dar tot filtrez distractorii evidenți.

**Memorie adaptivă:** moștenit din V5, cu aceleași valori (30/400).

### Rezultate finale pe hard-22

Câștiguri reale față de V3: seagull −0.220, motorcycle-18 −0.285, carchase −0.147.

Catastrofe față de V3: colorfish +0.498, motorcycle +0.470, racing-16 +0.228, crab-6 +0.156.

**Net dR V6 față de V3 = +0.056.** V3 câștigă.

### De ce nu merge V6

Diagnosticul post-hoc: automatul de absență cu `MEM_WINDOW_SHORT=30` distruge secvențele normale unde `obj_logit` e zgomotos sau obiectul trece printr-o ocluzie parțială scurtă. Fereastra scurtă șterge exact frame-urile care ar fi ajutat trackerul să se recupereze. Câștigurile pe secvențe cu absențe reale lungi (motorcycle-18, seagull) nu compensează regresiile pe secvențele fără absențe sau cu absențe scurte.

---

## Punctul de cotitură — feedback real de la server și calibrare pe Q real

Până aici toate deciziile s-au bazat pe metrica proxy `dR`. Lucram cu presupunerea că GT-ul competiției e ascuns permanent pe server și că nu primesc niciun răspuns înainte de deadline — deci proxy-ul era singura busolă. **Două descoperiri au schimbat complet imaginea.**

### Descoperirea 1: serverul ÎNTOARCE Q real înainte de deadline

Am submis baseline-ul (`dam4sam_b`) și serverul a răspuns cu scorul real pe setul de test public VOTS2026:

| Măsură | Valoare | Ce înseamnă |
|--------|---------|-------------|
| **Q** | **0.69** | scorul principal de calitate — ținta de bătut |
| Acc | 0.77 | acuratețea segmentării când urmărește |
| Rob | 0.86 | fracția urmărită fără eșec — deja mare |
| **DRE** | **0.09** | drift: prezice prezent dar greșit (distractor) |
| NRE | 0.05 | ratări: obiect prezent dar prezice gol |
| ADQ | 0.64 | — |

Presupunerea mea centrală era greșită. Și, mai important, **cele secundare sunt un diagnostic**: baseline-ul pierde aproape de două ori mai mult la **drift (DRE 0.09)** decât la **ratări (NRE 0.05)**. Asta îmi spune unde să țintesc — reducerea drift-ului — și unde NU (mecanismele de absență, fiindcă ratările sunt deja rare). Bonus: buget limitat de submisii (9 rămase pe acest nume), deci trebuie să filtrez local înainte de a trimite.

### Descoperirea 2: VOST-val ca oracol de Q real local

Am găsit stack-ul `vots2024/votstval` — **70 de secvențe cu GT public complet** (mască per-frame), singurul dataset pe care pot rula `vot analysis` și obține Q real local.

#### Ce este și ce nu este VOST-val față de VOTS2026

Un detaliu de structură important de pronunțat explicit: **VOST-val și VOTS2026 sunt dataset-uri complet diferite — zero secvențe în comun.**

| | **VOST-val** | **VOTS2026** |
|---|---|---|
| Secvențe | 70 | 170 |
| Origine | EPIC-Kitchens (bucătărie) | VOT benchmark (general) |
| Domeniu | 100% gătit: tăiat ceapă, frământat aluat, spart ouă | Animale, vehicule, sport, drone-uri, pești, insecte |
| Lungime medie | ~222 cadre | ~1859 cadre (**de 8× mai lungi**) |
| Lungime maximă | 455 cadre | 10699 cadre |
| Tip mișcare | **Lentă, deformabilă** (mâini, aluat, legume) | Variată: de la pești lenți la UAV-uri rapide |
| GT disponibil | **Public complet** (`groundtruth_object<N>.txt`) | **Secret** — Q real vine de la server |

VOTS2024 a fost o competiție separată; setul de validare public (`votstval`) l-au lăsat accesibil tocmai pentru benchmarking. Nu îl folosim pentru că ne place bucătăria — îl folosim pentru că e **singurul loc cu GT public** unde putem rula `vot analysis` și obține un Q real, nu un proxy.

#### Cum am obținut fiecare dataset

Ambele se descarcă prin `vot-toolkit` (`vot initialize <stack>`), care trage secvențele de pe serverul oficial VOT și le pune în workspace-ul indicat:

| Dataset | Comandă | Mărime pe disc | GT |
|---|---|---|---|
| **VOTS2026** | `vot initialize vots2026/main --workspace workspace2026` | ~5.0 GB (170 seq) | **Held-out permanent** — la descărcare vine doar masca de pe frame 0; restul rămâne pe server |
| **VOST-val** | `vot initialize vots2024/votstval --workspace workspace_vost_val` | ~5.7 GB (70 seq) | **Public complet** — mască per-frame + `evaluation.tag` + `groundtruth__ignore.txt` |

Diferența de GT nu e o alegere de design, ci natura competiției: pentru setul activ (VOTS2026) organizatorii țin adevărul ascuns ca să nu se poată antrena/regla pe el — de aici toată problema noastră de „nu am pe ce măsura local". VOST-val e un set *expirat* dintr-o ediție trecută, deci adevărul a fost publicat. (VOTS2023 are același regim held-out ca 2026; VOT2021/2022 sunt din era bbox, cu metrică diferită, deci inutilizabile pentru Q-pe-mască.)

**Consecința pentru interpretarea calibrării:** Q-ul de pe VOST-val are o părtinire sistematică față de VOTS2026 — reflectă performanța *pe mișcare lentă și deformabilă*, nu pe mișcare rapidă. Un tracker care câștigă pe VOST-val nu garantează câștig pe VOTS2026 și viceversa. Asta înseamnă că nu pot extrapola direct: „V3 pierde cu 0.011 pe VOST-val → V3 pierde și pe VOTS2026." Pot folosi VOST-val ca dovadă că *pe mișcare lentă* V3 strică ușor — ceea ce justifică V3g (poarta se închide pe mișcare lentă, revenind la baseline), dar nu îmi spune cât de mult câștigă V3g pe secvențele rapide din VOTS2026.

**Notă de metodologie care m-a salvat de o noapte pierdută:** înainte de rularea pe 70×3 trackere, am făcut un smoke-test pe O singură secvență. A prins un bug de parsare (fișierele VOST-val se numesc `groundtruth_object<N>.txt`, nu `_obj<N>` — runner-ul standalone le-ar fi sărit pe toate în tăcere) și a confirmat că pipeline-ul oficial dă Q real. Lecția: validează lanțul de evaluare pe o unitate înainte de orice rulare lungă.

### Rezultatul calibrării — și o corecție onestă

Baseline, V3 și V6 pe toate cele 70, cu `vot analysis` oficial (69 secvențe comune):

| Tracker | **Q real** | Robustness |
|---------|-----------|------------|
| **baseline** | **0.392** | 0.731 |
| V3 | 0.381 | 0.704 |
| V6 | 0.371 | 0.682 |

**Pe Q real, baseline câștigă. V3 și V6 sunt amândouă ușor mai slabe.** A fost un duș rece — credeam că V3 e clar mai bun.

Am verificat dacă proxy-ul mă mințea. Am calculat proxy-ul `R` pe **același** dataset: baseline 0.291 < V3 0.295 < V6 0.354 — **același clasament ca Q real**. Deci proxy-ul NU minte: pe același domeniu urmărește corect Q real. Greșeala mea fusese alta: „V3 mai bun" venea din proxy-ul pe **cele 22 secvențe grele** (mișcare rapidă), un alt set de date. Discrepanța era între **domenii**, nu între proxy și adevăr.

### Ce am înțeles de fapt

1. **Axa absenței (V4–V6) e moartă pentru VOTS2026.** Serverul arată NRE deja mic (0.05). Iar VOST-val — domeniul cu cea mai grea absență, exact unde V6 ar fi trebuit să strălucească — îl arată *mai slab* ca baseline. Dacă mecanismul de absență ar fi ajutat undeva, ar fi ajutat acolo. Renunț la el.

2. **V3 e o unealtă specifică pe mișcare rapidă.** Pe obiecte lente/deformabile modelul Kalman constant-velocity prezice greșit și V3 devine marginal dăunător. V3 brut aplică motion-ul *oricând* SAM e ambiguu — fără să verifice dacă mișcarea e informativă.

3. **Ținta corectă e drift-ul (DRE), atacat conservator** — exact slăbiciunea baseline-ului, exact ce face V3, dar numai unde chiar ajută.

---

## V3g — Motion cu Poartă de Viteză

**Fișier:** [dam4sam/dam4sam_tracker_v3g.py](dam4sam/dam4sam_tracker_v3g.py)

V3g e răspunsul direct la diagnostic. Repară defectul lui V3: pe lângă condiția existentă (SAM ambiguu + KF sănătos), adaugă o **poartă de viteză** — motion-ul se aplică numai dacă obiectul se mișcă suficient de repede încât mișcarea să fie informativă.

Pragul e **scale-invariant** — viteza centroidului raportată la dimensiunea obiectului:

```
speed_ratio = |(vx, vy)| / sqrt(w·h)
poarta deschisă  ⇔  speed_ratio ≥ VEL_RATIO_THR   (start: 0.03)
```

Independent de rezoluție și de mărimea obiectului. Pe obiecte lente (kitchen) `speed_ratio ≈ 0` → poarta închisă → `argmax(sam_iou)` pur = comportament Base+. Pe mișcare rapidă (uav, racing, seagull) → poarta deschisă → motion reduce drift-ul, ca în V3.

**Proprietatea cheie, prin construcție:** când poarta e închisă, V3g se comportă identic cu baseline-ul → **nu poate fi mai rău acolo unde V3 era mai rău**. Țintește DRE-ul fără să atingă secvențele lente. Implementat ca subclasă a V3 care suprascrie un singur hook (`_motion_gate_open`); V3 a primit hook-ul neutru (întoarce mereu `True`), deci comportamentul lui rămâne neschimbat.

### Validarea pe proxy: poarta se comportă exact cum a fost proiectată

Am testat V3g pe 6 secvențe alese ca să verific cele două comportamente cerute: pe mișcare rapidă (unde V3 câștigă) poarta trebuie să rămână deschisă; pe secvențele unde V3 strica, poarta trebuie să se închidă și să revină la baseline. `dR` negativ = mai bun ca baseline.

| Secvență | Tip | base R | V3 (dR) | **V3g (dR)** | Verdict poartă |
|----------|-----|--------|---------|--------------|----------------|
| seagull | mișcare rapidă | 0.958 | −0.168 | **−0.433** | deschisă — câștig amplificat |
| racing-16 | mișcare rapidă | 0.448 | −0.254 | **−0.258** | deschisă — câștig păstrat |
| fish2 | distractori | 0.829 | +0.002 | **+0.002** | identic V3 |
| babychimp | lentă/deformabilă | 0.003 | (regresie V3) | **+0.000** | închisă — exact baseline |
| rabbit | absență lungă | 1.013 | −0.006 | **−0.006** | identic |
| motorcycle-18 | absență + re-achiziție | 0.567 | **+0.302** ❌ | **−0.431** | închisă la KF stale — recuperare |

**Zero regresii. V3g e ≥ baseline pe toate 6.** Rezultatul-cheie e motorcycle-18: V3 brut *strica* secvența (+0.302, KF-ul rămânea cu viteză veche după absență și selecta masca greșită la re-achiziție); V3g detectează că viteza KF nu e credibilă, închide poarta, și SAM re-detectează singur corect → câștig de −0.431. Net pe cele 6: V3g ≈ −1.13 vs V3 ≈ −0.12.

### Validarea finală: Q real pe VOST-val (69 secvențe comune, `vot analysis` oficial)

| Tracker | **Q real** | Robustness | Accuracy | NRE | DRE |
|---------|-----------|------------|----------|-----|-----|
| **baseline** | **0.392** | 0.731 | 0.422 | 0.120 | 0.149 |
| **V3g** | **0.388** | 0.727 | 0.420 | 0.120 | 0.154 |
| V3 | 0.381 | 0.704 | 0.413 | 0.128 | 0.168 |
| V6 | 0.371 | 0.682 | 0.403 | 0.159 | 0.159 |

**V3g Q=0.388 — cel mai bun din toate variantele noastre, la 0.004 de baseline.**

Citind tabelul: V3g reduce complet regresia lui V3 (care era −0.011 față de baseline; V3g e −0.004). NRE-ul (ratările) e identic cu baseline (0.120 = 0.120) — poarta nu introduce false absențe. DRE (drift) e ușor crescut (+0.005) față de baseline — explică cei 0.004 în Q: ocazional, la viteze la limita pragului, poarta se deschide pe un KF marginal și mai introduce o selecție greșită.

Condiția strictă „≥ baseline (0.392)" nu e atinsă cu 0.4 puncte procentuale, *pe cel mai defavorabil domeniu* pentru V3g (bucătărie = mișcare lentă, unde poarta ar trebui să rămână închisă aproape tot timpul). Pe secvențele rapide din VOTS2026 — exact unde DRE e pierderea dominantă — V3g câștigă masiv (proxy: seagull −0.433, racing −0.258, motorcycle-18 −0.431). VOTS2026 conține ambele tipuri, deci V3g rămâne candidatul principal.

---

## DAM4SAM-L — Upgrade de Backbone la SAM2.1-Large

**Fișiere:** [dam4sam/vot_wrapper_large.py](dam4sam/vot_wrapper_large.py), `workspace2026/trackers.ini` (`dam4sam_l_cloud`)

### Motivație

Câștigătorul VOTS2025 (S3-DAM4SAM, Q=0.751) a folosit un backbone SAM2.1 de dimensiune mare. DAM4SAM original și toate variantele noastre V2–V3g rulează pe **SAM2.1-Base+** (Hiera-B+). Upgrade-ul la **SAM2.1-Large** (Hiera-L) — de ~3× mai mare ca encoder — este schimbarea cu cel mai mare raport beneficiu/risc de cod: zero modificare la logica DRM sau de tracking, doar swap de checkpoint.

Blocked local: Hiera-L necesită ~5–6 GB VRAM static + memorie pentru stările de cadru → depășește 10 GB-ul RTX 3080-ului. Soluție: cloud GPU (vast.ai, RTX 4090 24 GB).

### Ce am implementat

`vot_wrapper_large.py` este identic cu `vot_wrapper_multiobject.py` cu două diferențe:
1. Încarcă `sam2.1_hiera_large.pt` în loc de checkpoint-ul Base+.
2. Suprascrie `init_state_tw()` să seteze `storage_device=cpu` — stările de cadru se stochează pe CPU între frame-uri, prevenind OOM pe secvențe de mii de frame-uri cu 24 GB VRAM.

### Rularea pe cloud — infrastructură și incidente

Am închiriat **2× RTX 4090 (24 GB)** pe vast.ai (~$0.50/h fiecare). Split-ul celor 170 de secvențe s-a făcut **după numărul de frame-uri** (nu după numărul de secvențe), ca workload-ul GPU să fie echilibrat:

| Instanță | Secvențe | Frame-uri | Durată estimată |
|----------|----------|-----------|-----------------|
| Bulgaria (108016) | 78 | ~echilibrat | ~18h |
| Finland (151177) | 72 | ~echilibrat | ~18h |
| Local 3080 (asigurare) | 31 (cele mai scurte) | ~14% din total | paralel |

**Incident major — GPU Xid fault:** Prima instanță (38.117.87.43:40913) s-a stricat hardware la jumătatea rulării — `nvidia-smi` nu mai raporta GPU-ul. Lecție: SSH rămâne deschis chiar și după un fault hardware → am putut face `rsync` și salva cele 20 de rezultate existente înainte de a distruge instanța.

**Consecință:** Am creat `pull_cloud_results.sh` — un script local care face `rsync` incremental la fiecare 3 minute cât timp rularea e activă. Non-negociabil pentru orice rulare pe cloud în viitor.

**Bug `iopath` lipsă:** Ambele instanțe fresh au dat `ModuleNotFoundError: no module named 'iopath'` la primul smoke-test. Cauza: `cloud_setup.sh` nu instala pachetul. Fix: `uv pip install iopath`. De adăugat în `cloud_setup.sh`.

### Rezultat final

170/170 secvențe verificate strict (`check_results_integrity.py --strict`): **170 OK / 0 pending / 0 FAILED**. Arhivă: `workspace2026/dam4sam_l_cloud_2026-06-13T03-12-07.052254.zip` (364 MB, integritate `zipfile.testzip() = None`).

Câștig estimat față de B+: **+0.02–0.04 Q** pe baza corespondentei din VOTS2025 (câștigătorul cu L = Q=0.751 vs B+ = Q mai mic). Răspunsul real vine de la server după submisie.

---

## D2 — DRM-Admission Gating (canal de identitate, direcție validată pe LaSOT)

**Fișier:** [dam4sam/dam4sam_tracker_v7.py](dam4sam/dam4sam_tracker_v7.py) (flag `USE_D2`)

### Motivație

Diagnosticul server (DRE=0.09 >> NRE=0.05) arată că pierderea principală e **drift** — trackerul alege un distractor și îl urmărește în loc de obiectul real. Câștigătorul VOTS2025 a redus DRE de la 0.09 la 0.04 cu un canal de identitate independent (Cutie), gated. Ideea: nu poți recombina output-urile SAM2 (un distractor care păcălește IoU head îl păcălește și pe `obj_logit` — corelate); ai nevoie de un semnal de identitate **independent** de deciziile SAM2.

### Ce am implementat

**Signal extraction (zero risc comportamental — read-only):**
- Forward hook pe `self.sam_mask_decoder` capturează `obj_ptr` per-candidat (nativ SAM2 expune doar token-ul best-IoU; hook-ul citește toți tokenii și aplică `obj_ptr_proj`).
- Canal 2: mask-pooling pe `backbone_fpn` stride-16 (features cached, zero recompute) → L2-normalize → cosine similarity.

**Prototip bank (rezolvarea chicken-and-egg):**
- `P0` = embedding din frame 0, imuabil permanent (leash anchor).
- Slot rulant = ultimul frame confident (`iou > 0.8`, tie închis, fără flag `suspect`).
- Admitere strictă: `cos(emb, P0) > LEASH_TAU` + gap minim + probation. Adaptivitatea în scor (`max over bank`), siguranța în admitere (conjuncție strictă).

**DRM gating:** `add_to_drm` interceptat — frame-ul e blocat din memorie dacă `id_score < TAU_DROP` sau `suspect[t]` e setat.

### Validare pe LaSOT (12 secvențe, 26 onset-uri drift, GT real)

Am descărcat 5 categorii LaSOT (bird/monkey/zebra/motorcycle/drone, ~19 GB) ca set de drift cu GT real — dizolvând capcana domeniului bucătărie (VOST-val). 8 din cele 170 VOTS2026 se suprapun cu LaSOT-test → excluse din harness.

Rezultat pe drone-13 (8 eventi drift la baseline, 5 la V3g):
- V7-D2 (DRM-gating only, mem-drop dezactivat): **4 eventi** — bate V3g fără regresii.
- Mem-drop era nociv (lecția rabbit/V6); D2 pur = admitere mai strictă în DRM, fără drop activ.

Calibrare id-score: curate p50=0.99, drift p50=0.59 → prag 0.7 = 98% din admiterea curată păstrată / 79% din admiterea la drift blocată.

**Bit-identical cu V3g flags-off** — verificat pe MD5 pe 2 secvențe.

Statut: **candidat pentru S3** (combinat cu chassis-ul cel mai bun după S1/S2).

---

## Rezultate reale de pe server + ablație

Serverul VOTS2026 întoarce Q real prin email (submisii limitate, ~1/zi). Ce am trimis și ce a răspuns:

| Submisie | Tracker | Q | Acc | Rob | NRE | DRE | ADQ | Verdict |
|---|---|---|---|---|---|---|---|---|
| referință | `dam4sam_b` (B+) | 0.69 | 0.77 | 0.86 | 0.05 | 0.09 | 0.64 | ținta de bătut |
| #1 | `dam4sam_v3g` (B+ +motion) | 0.69 | 0.79 | 0.87 | 0.03 | 0.09 | 0.62 | no-op pe Q |
| #2 | `dam4sam_l_cloud` (Large) | **0.72** | 0.80 | 0.88 | 0.05 | 0.08 | 0.74 | **backbone bate baseline** (+0.03) |
| #3 | `dam4sam_l_d2` (Large +motion +D2) | 0.71 | 0.80 | 0.87 | 0.04 | 0.09 | 0.65 | regresie vs L (−0.01) |
| #4 | `dam4sam_v7d2` (B+ +motion +D2) | 0.69 | 0.79 | 0.87 | 0.04 | 0.10 | 0.63 | no-op pe Q (D2 izolat, in-calibration) |

### Ce a spus fiecare rezultat

**#1 — V3g (poarta de mișcare) e Q-neutră.** Ipoteza era „taie drift-ul (DRE)". A eșuat exact pe ținta ei: **DRE 0.09 → 0.09, neclintit.** A redus ratările (NRE 0.05→0.03) și a crescut puțin acuratețea, dar pe primar e wash total (Q 0.69 = 0.69). Lecția: drift-ul nu se sparge cu un model de mișcare; motion-ul ajută pe secvențe rapide, dar pe tot setul de 170 (multe lente) câștigurile și pierderile se anulează. `results.json` local reproduce exact emailul (Q=0.6913) → pipeline-ul de evaluare e calibrat pe server.

**#2 — Backbone-ul Large câștigă pariul singur.** L = **Q=0.72**, +0.03 față de baseline, exact la nivel stretch. Câștigul vine din **ADQ +0.10** (0.64→0.74), Acc +0.03, Rob +0.02, și chiar puțin DRE (0.09→0.08). Confirmă lecția #3: un swap de backbone bate luni de tunare a logicii. Câștigătorul VOTS2025 (Q=0.751) folosea tot un backbone mare. **Pariul de 200€ e câștigat de aici.**

**#3 — L+D2 a REGRESAT (0.71 < 0.72).** Speranța era ca D2 să prăbușească drift-ul real al lui L (cum a făcut cu `drift_frac` de 4× pe drone-13). În schimb **D2 a făcut opusul: DRE a crescut** (0.08→0.09) și ADQ s-a prăbușit (0.74→0.65). Semnalul de pe LaSOT/drone-13 a fost înșelător — nu s-a transferat pe setul real multi-obiect. *(Infrastructură: rulat pe 2× RTX 4090 cloud; boxurile au murit la epuizarea bugetului dar `pull_cloud_results.sh` salvase 165/170, restul completat local; integrity check strict a prins o secvență cu un obiect trunchiat de `rsync --partial` înainte de submisie.)*

**#4 — B++D2 izolat e Q-neutru (0.69 = baseline).** Acesta e testul *in-calibration*: D2 pe ACELAȘI backbone (B+) ca baseline-ul, cu pragurile native B+. Verdictul e decisiv: **chiar și in-calibration, D2 nu adaugă valoare** — Q egal, DRE chiar ușor mai rău (0.10). Asta răstoarnă ipoteza recalibrării: problema NU e că pragul era mis-setat pe Large, ci **semnalul de identitate însuși** — nu distinge fiabil „schimbare legitimă de aspect" de „distractor" pe date reale multi-obiect, deci blochează cadrele greșite din DRM. Pe single-obiect LaSOT părea că merge; pe 170 multi-obiect, nu.

### Tabelul de ablație (backbone × canal de identitate) — Q real, complet

|  | **B+ backbone** | **Large backbone** | **efect backbone** |
|---|---|---|---|
| **base** | 0.69 | **0.72** | **+0.03** |
| **+ D2 identitate** | 0.69 | 0.71 | +0.02 |
| **efect D2** | **0.00** | **−0.01** | |

Izolare completă, toate celulele cu Q real de pe server. Citirea e fără echivoc:
- **Backbone-ul (B+→Large) = +0.03** — singurul levier care mișcă acul.
- **D2 (canalul de identitate) = 0.00 pe B+, −0.01 pe L** — nu adaugă valoare pe niciun backbone.

Faptul că **D2 e neutru chiar și in-calibration (pe B+, pragurile lui native)** e cheia: elimină ipoteza „doar prost calibrat pe Large". Semnalul de identitate construit de noi nu separă fiabil obiectul de distractor pe datele reale multi-obiect — deci gating-ul de memorie blochează cadre greșite. Pe LaSOT single-obiect (drone-13) părea promițător (drift_frac 4×), dar a fost o capcană de extrapolare între domenii — exact riscul din secțiunea de metodologie.

---

## Metodologia de evaluare — ce am învățat să măsor

În tracking, **media ascunde eșecurile**: o creștere mică de Q mediu poate masca câteva regresii catastrofale. Principiile de măsurare pe care le-am adoptat (parte din ele după feedback extern, parte din propriile capcane):

1. **Regresia worst-case e metrică de prim rang**, nu doar media. Cele mai grave decizii — babychimp +0.409 la V3, rabbit +1.36 la V2 — au fost prinse uitându-mă la *cea mai mare pierdere per secvență*, nu la medie. Obiectivul nu e doar să crești media, ci să **reduci regresiile rare dar catastrofale** (`run_eval_170.py` are `--max-regression` / `--max-catastrophes` exact pentru asta).

2. **Stratificare pe tip de dificultate**, nu un singur scor agregat. Aceleași date, văzute pe categorii: mișcare rapidă / obiecte lente / cameră mobilă / distractori / deformare / ocluzii / absențe lungi. Asta a dezvăluit că V3 e bun pe mișcare rapidă dar dăunător pe lente — invizibil în media globală. De aici V3g (poarta care se închide pe lent).

3. **Win/loss per secvență + mediana diferențelor**, nu doar suma. `compare_robustness.py` raportează câte secvențe câștigă/pierde fiecare variantă — un net pozitiv din 2 câștiguri mari + 10 pierderi mici e fragil.

4. **Harta domeniilor de date** (nu extrapola un set la altul): hard-22 = screening mișcare rapidă; VOST-val Q = regresie pe domeniu lent (0 overlap cu cele 170, bucătărie); LaSOT-drift = validare drift cu GT real. Un câștig pe un domeniu NU garantează câștig pe altul (lecția centrală a calibrării V3).

## Direcții viitoare (identificate, ne-implementate)

Idei validate ca direcție corectă, lăsate pentru un „v2" post-competiție:

1. **Poarta de mișcare pe încredere, nu pe viteză brută.** Poarta V3g deschide pe `speed_ratio ≥ 0.03` — dar un obiect rapid nu e neapărat *predictibil*. Filtrul Kalman dă gratis inovația (predicție vs observație) normalizată de covarianță; o poartă pe consistența predicției (NIS / distanță Mahalanobis) e strict mai discriminantă și rezolvă parțial și problema camerei.

2. **Compensarea mișcării de cameră (ego-motion).** Viteza centroidului în imagine include și deplasarea camerei (UAV, racing, handheld) → poarta V3g se poate deschide degeaba. Fix: estimează mișcarea globală (optical flow / keypoint-uri de fundal), scade-o, obține mișcarea relativă la scenă. Cost mare (per-cadru) → de aceea „viitor".

3. **Model de mișcare IMM** (Interacting Multiple Models) în loc de viteză constantă — păstrează în paralel modele static / viteză-constantă / accelerare și estimează care explică mișcarea curentă. Mai robust pe cazuri ca motorcycle-18 (KF blocat pe viteză veche după o absență).

4. **Prototipuri negative din candidatele runner-up ale SAM2.** D2 actual gating-uiește pe similaritate cu *ținta*; un canal negativ (e masca aleasă mai aproape de un *distractor* cunoscut?) ar întări veto-ul de identitate. Sursa naturală: măștile alternative SAM2 cu scor apropiat de cea aleasă.

---

## Concluzie

Direcția s-a clarificat abia după ce am avut Q real. Lecția metodologică de bază rămâne — *e mai ușor să strici un tracker bun decât să-l îmbunătățești* — iar V3 a respectat principiul „nu interveni dacă nu e necesar" pe care V4–V6 l-au încălcat. Feedback-ul real a adăugat un strat: **proxy-ul e fidel pe domeniu, dar un câștig pe proxy pe un subset nu garantează un câștig net pe Q real pe tot setul.** De aici V3g — care duce principiul mai departe: nu interveni decât unde ai dovada că mișcarea contează.

Pe parcurs, analiza DRE/NRE a dat o a doua lecție: **axa absenței e moartă pentru VOTS2026** (NRE deja mic) — tot efortul V4–V6 a fost cheltuit pe direcția greșită. Calibrarea pe VOST-val a clarificat asta: V6 (cel mai sofisticat mecanism de absență) e mai slab ca baseline chiar pe domeniul cu cele mai multe absențe.

A treia lecție: **backbone-ul contează mai mult decât logica de tracking pe termen scurt.** Un swap de checkpoint (B+ → Large) la zero risc de cod e probabil mai valoros decât luni de tunare a pragurilor — câștigătorul VOTS2025 demonstrează asta cu Q=0.751.

**Verdictul final (2026-06-19), cu ablația completă pe Q real:**

Ablația 2×2 e fără echivoc: **singurul levier care mișcă Q e backbone-ul** (B+→Large = +0.03). Ambele contribuții algoritmice făcute de mână — poarta de mișcare (V3g) și canalul de identitate (D2) — sunt **Q-neutre sau ușor negative pe Q real**, pe orice backbone. D2 a fost testat și *in-calibration* (pe B+, pragurile lui native) și tot n-a adăugat valoare → problema nu e calibrarea, e că semnalul de identitate construit de noi nu se transferă de pe LaSOT single-obiect pe setul real multi-obiect.

Onest despre pariul de 200€: în sensul strict — *o modificare algoritmică de-a noastră bate baseline-ul pe Q real* — **nu a fost atins.** E un rezultat negativ real, obținut prin testare riguroasă, nu printr-o eroare de execuție. Cea mai bună intrare rămâne **DAM4SAM-L = Q=0.72** (peste baseline 0.69): un câștig de configurație + inginerie (backbone mare rulat prin offload VRAM + infra cloud), legitim ca intrare de competiție, dar nu o contribuție algoritmică originală.

Valoarea reală a proiectului nu e scorul, ci **rigoarea negativă**: am demonstrat curat, cu oracol real, că pe VOTS2026 logica training-free făcută de mână (re-ranking de mișcare, gating de identitate) nu mișcă acul — câștigă scala modelului. Plus un catalog de capcane (extrapolare între domenii, proxy vs Q real, over-blocking de memorie) care sunt ele însele lecții transferabile. Deadline oficial: 22 iunie.

### Notă istorică (înainte de Q real)

Versiunea inițială a acestui jurnal conchidea că **V3 e cel mai bun candidat**, pe baza proxy-ului `dR`. Păstrez observația ca etapă de raționament: era corectă pe datele pe care le aveam atunci, dar calibrarea pe Q real a nuanțat-o (vezi secțiunea de mai sus).

Cel mai important lucru pe care l-am învățat: e mai ușor să strici un tracker bun decât să-l îmbunătățești. DAM4SAM e deja foarte bun pe secvențele normale — orice adăugare care intervine și pe frame-urile fără probleme va da regresii acolo. V3 a rezolvat asta cu principiul „nu interveni dacă nu e necesar". Versiunile V4–V6 au intervenit mai agresiv și au plătit prețul.

---

## Diagrama de moștenire

```
DAM4SAMTracker  (baza SAM2.1-B+)
│
├── DAM4SAMTrackerV2        blend motion global — net zero, înțeles de ce eșuează
│
├── DAM4SAMTrackerV3        tiebreaker + KF lifecycle + postproc condiționat
│   ├── DAM4SAMTrackerV3b   același cod, TIE_MARGIN 0.15→0.05
│   └── DAM4SAMTrackerV3g     + poartă de viteză (motion gated)      ← #1 trimis: Q=0.69 (no-op)
│       └── DAM4SAMTrackerV7 (USE_D2) + DRM-admission gating
│           ├── pe B+ → B++D2 (rulat local, ablație)
│           └── pe Large → L+D2                                      ← #3 trimis (în așteptare)
│
├── DAM4SAMTrackerV4        automat PRESENT/ABSENT — bug declarare falsă
│   ├── DAM4SAMTrackerV5    fereastră adaptivă — SHORT=30 distruge ancorele
│   └── DAM4SAMTrackerV6    declarare conservatoare — câștig pe absențe reale,
│                           regresii pe secvențele normale
│
└── [cloud] DAM4SAM-L       SAM2.1-Large backbone, logică DRM identică ← #2 trimis: Q=0.72 ✅ PARIU
    (vot_wrapper_large.py)  CPU offload frame states; 170/170 pe 2× RTX 4090
```

Fiecare versiune e un subclass pur — `dam4sam_tracker.py` nu a fost modificat niciodată direct.

---

## Tabel rezumativ

| Versiune | Ipoteza testată | Rezultat | Lecție |
|----------|-----------------|----------|--------|
| Base | DAM4SAM original (B+) | referință Q=0.69 server | — |
| V2 | blend KF global ajută selecția măștii | net ~0 | blend global otrăvește KF pe frame-uri neambigue |
| **V3** | KF numai ca tiebreaker + lifecycle corect | pozitiv pe proxy; **sub baseline pe Q real** | proxy fidel pe-domeniu, dar câștig pe subset ≠ câștig net |
| V3b | TIE_MARGIN mai strict previne derivarea | microfix | un parametru mai mic elimină un outlier catastrofal |
| V4 | automat de absență cu 3 semnale | negativ — false absențe | APP_SIM și m_iou prea zgomotoase în declarare |
| V5 | fereastră de memorie legată de stare | parțial — nou bug | SHORT=30 distruge ancorele înainte de comutare |
| V6 | declarare numai pe obj_logit nativ SAM2 | **sub baseline pe Q real** (VOST-val) | axa absenței moartă: NRE deja mic (0.05) |
| **V3g** ✅ | motion DOAR pe mișcare rapidă (poartă de viteză) | **Q=0.388** (VOST-val); câștig masiv pe rapide | cel mai bun pe B+; la 0.004 de baseline pe domeniu lent |
| **DAM4SAM-L** | backbone SAM2.1-Large izolat (cloud 2×4090) | **Q=0.72 real** (+0.03 față de B+) — cea mai bună intrare | backbone = singurul levier care mișcă Q; câștigătorul VOTS2025 a folosit L |
| **L+D2** | Large + poartă mișcare + DRM-admission gating pe identitate | **Q=0.71 — regresie** vs L; DRE crescut | semnalul LaSOT (drift_frac 4×) NU s-a transferat pe 170 multi-obiect |
| **B++D2** | D2 izolat pe B+ (in-calibration) | **Q=0.69 — neutru** = baseline | D2 nu adaugă valoare nici in-calibration → semnalul de identitate e problema, nu pragul |

**Diagnostic server (baseline Q=0.69):** DRE 0.09 (drift) ≫ NRE 0.05 (ratări) → ținta e drift-ul. Dar nici motion (V3g) nici identitate (D2) nu au redus DRE pe Q real — drift-ul a rămas ne-spart de logica training-free.

**Ablație finală (Q real):** backbone B+→Large **+0.03**; D2 **0.00 pe B+ / −0.01 pe L**. **Singurul levier real = scala backbone-ului.** Pariul (contribuție algoritmică > baseline) NU atins; cea mai bună intrare = L=0.72 (configurație, nu contribuție originală).

**Submisii pe server:**  #1 V3g → 0.69 (no-op) · #2 L → **0.72** (best) · #3 L+D2 → 0.71 (regresie) · #4 B++D2 → 0.69 (D2 izolat, neutru). Deadline 22-iun.

---

*Actualizat: 2026-06-19 (ablație 2×2 COMPLETĂ pe Q real: backbone +0.03 singurul levier; D2 neutru pe B+ și −0.01 pe L → contribuțiile training-free nu bat baseline; verdict final onest)*
