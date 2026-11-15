# Breakout: DQN i Double DQN

Cztery pliki w `experiments/ConceptCheck`: `model.py` — CNN, `train.py` — uczenie,
`rewards.py` — środowisko i nagrody, `evaluate.py` — ocena. Testy są w jednym pliku.

Instalacja: Python 3.12, `python -m pip install -r requirements.txt`.
GPU wymaga PyTorch z CUDA; bez GPU podaj `--device cpu`.

```powershell
python experiments/ConceptCheck/train.py --algorithm double-dqn --output outputs/ddqn --decisions 15000000 --device cuda --seed 230849415
python experiments/ConceptCheck/evaluate.py --model outputs/ddqn/model.pt --output outputs/ddqn/eval.json --episodes 100 --device cuda
python -m unittest discover -s tests -v
```

`--algorithm dqn` wybiera zwykły DQN. Obie metody mają tę samą sieć: cztery szare
klatki 84×84, trzy konwolucje i dwie warstwy liniowe. Nie ma PER, n-step ani dueling.
Domyślnie co milion decyzji zapisujemy ocenę 100 gier i `comparison.csv`.
`--eval-every 0` wyłącza oceny. Jedna decyzja obejmuje domyślnie cztery klatki gry.

Parametry: `--lr`, `--gamma`, `--buffer-capacity`, `--train-freq`, `--target-update-freq`,
`--batch-size`, `--min-replay-size`, `--mode`, `--difficulty`, `--frame-skip`
i `--sticky-action-probability`. Bez seedu losujemy go i zapisujemy w modelu.

`--reward` wybiera jeden z trzech wariantów:
- `raw`: wyłącznie prawdziwe punkty za cegły; domyślny wariant.
- `small-bounce`: dodatkowo 0,05 za odbicie i 0,01 za punkt, najwyżej 0,10 bonusu na wymianę.
- `landing`: dodatkowo `0,1 × (gamma × Phi_następne − Phi_obecne)`; Phi mierzy ustawienie
  paletki pod przewidywanym miejscem opadania piłki, wykrywanym z obrazu.

Bonusy są eksperymentalną podpowiedzią treningową. Oceny zawsze podają prawdziwe
punkty gry. CNN dostaje wyłącznie obrazy, bez współrzędnych i RAM gry.

`model.pt` zawiera sieci, Adam, konfigurację i RNG. `--start model.pt` kontynuuje
z tych wag, ale zaczyna nową grę i pusty bufor — to nie jest dokładne wznowienie.
Prosty bufor 500 tys. doświadczeń zajmuje około 14 GB RAM. Używaj osobnych katalogów.
Poprzedni kod i wyniki zachowano lokalnie w `outputs/archive/`; wcześniejsze raporty
pozostają w `reports/`. Nowe wnioski i wyniki prób zapisujemy lokalnie w `outputs/`.
