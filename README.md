# Reinforcement learning — Breakout

DQN i Double DQN w PyTorch, ze wspólną siecią CNN i pętlą treningową.

## Instalacja

Python 3.12, polecenia z katalogu repozytorium:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Opcjonalnie PyTorch z CUDA dla GPU NVIDIA (wymaga `uv`):

```powershell
uv pip install --python .venv\Scripts\python.exe --index https://download.pytorch.org/whl/cu132 --reinstall-package torch torch==2.15.0+cu132
```

## Uruchomienie

```powershell
python -m experiments.ConceptCheck.validate_stability --run-dir outputs/validation-new --device cuda --seed 230849415 --comparison-rate 150
python -m experiments.ConceptCheck.experiment --algorithm double-dqn --device cuda --seed 230849415 --eval-seed 3303497698 --max-steps-per-second 220 --run-dir outputs/ddqn-stable-15m
python -m experiments.ConceptCheck.experiment --algorithm dqn --device cuda --seed 230849415 --eval-seed 3303497698 --max-steps-per-second 0 --run-dir outputs/dqn-stable-15m
python -m experiments.ConceptCheck.experiment --run-dir outputs/ddqn-stable-15m --resume-experiment
python -m experiments.ConceptCheck.experiment --run-dir outputs/ddqn-stable-15m --resume-experiment --append-milestones 20000000,25000000 --max-steps-per-second 0
python -m experiments.ConceptCheck.test --model-path outputs/ddqn-stable-15m/model.step-010000000.pth --episodes 100 --seed 3303497698
python -m experiments.ConceptCheck.pilot --run-dir outputs/ddqn-improvements-pilot
python -m experiments.ConceptCheck.pilot --run-dir outputs/ddqn-improvements-pilot --resume
$milestones = (1..50 | ForEach-Object { $_ * 1000000 }) -join ','
python -m experiments.ConceptCheck.experiment --run-dir outputs/ddqn-combined-50m --milestones $milestones --replay-storage frames --terminal-on-life-loss --buffer-capacity 200000 --epsilon-decay-steps 1000000 --max-steps-per-second 0
python -m experiments.ConceptCheck.experiment --run-dir outputs/ddqn-revision-2 --milestones $milestones --algorithm double-dqn --lr 0.00005 --buffer-capacity 500000 --gamma 0.995 --train-freq 2 --reward-shaping rally --bounce-bonus 1 --rally-bonus-step 0.2 --rally-bonus-max-multiplier 2 --replay-storage frames --terminal-on-life-loss --epsilon-decay-steps 1000000 --max-steps-per-second 0
python -m experiments.ConceptCheck.metrics --run-dir outputs/ddqn-combined-50m --output-prefix reports/ddqn-combined-50m-metrics
python -m unittest discover -s tests -v
```

`--algorithm dqn` wybiera zwykły DQN. Sterownik domyślnie używa CUDA;
`train.py` i `test.py` wybierają sprzęt automatycznie, gdy pominięto `--device`.
Bez `--seed` ziarno jest losowane; jego wartość trafia do logu i konfiguracji treningu.
Sterownik ocenia 100 gier po 1, 2, 4, 7, 10 i 15 mln decyzji, zapisuje porównanie
i sam wznawia trening. Limit tempa zostawia przerwy dla innych aplikacji; `0` go usuwa.
Po zakończeniu biegu `--append-milestones` dopisuje dalsze oceny i kontynuuje pełny stan.
W `train.py` `--steps` to łączny cel, a `--stop-at` robi przerwę w treningu.
`.training.pt` zachowuje sieci, optymalizator, replay, RNG i stan gry. Wznawiaj tylko
zaufane lokalne pliki. Starsze `.step-*.pth` są małymi kopiami do oceny; pełny stan
jest nadpisywany. Dawne pliki samych wag nie zawierają pełnego stanu treningu.
Przy sticky actions w ALE 0.12.1 zapis nie gwarantuje identycznej trajektorii po
wznowieniu: emulator nie zapisuje całej historii ostatniego wejścia.
DQN i DDQN porównuj przy tych samych seedach, ustawieniach gry i liczbie decyzji,
w osobnych katalogach. Jeden krok to domyślnie około czterech klatek emulatora.
Nowe treningi używają LeakyReLU i deterministycznych operacji; obie metody muszą
mieć te same ustawienia. Stare modele zachowują ReLU. Kopiuj wagi razem z `.config.json`.
`model.health.json`/`.jsonl` pokazują cechy obrazu i gradienty; wykryte zapadnięcie
zatrzymuje trening z zapisem diagnostycznym. Krótki test nie gwarantuje wyniku po 15 mln.

Pilot uruchamia kolejno cztery Double DQN po 2 mln decyzji, z oceną 100 pełnych gier
przy 1 i 2 mln: kontrolny, maskowanie celu TD po utracie życia, bufor 200 tys.
i spadek epsilona przez 1 mln. Każdy zmienia jedną rzecz i używa wspólnego seedu
oraz oszczędnego bufora klatek (`--replay-storage frames`). Wyniki trafiają do
`comparison.csv`; to wstępna selekcja ustawień, a nie dowód przewagi z jednego treningu.
Wariant `ddqn-combined-50m` łączy trzy zmiany w jednym Double DQN. Każdy milion
decyzji kończy się oceną 100 gier; wynik poprzedniego treningu jest w `reports/`.
`metrics` zbiera ukończone oceny co milion i zapisuje CSV oraz wykres PNG/SVG.
Pokazuje średnią, medianę, maksimum i przedział 25–75% wyników gier.

Rewizja 2 zmienia LR, bufor, gamma, częstotliwość uczenia i dodaje ograniczony bonus.
Odbicie paletką daje `1`; punkty zdobyte do następnego odbicia podnoszą sumę bonusu
o `0,2` za punkt, najwyżej do `2`. Dopłacana jest tylko różnica. Prawdziwe punkty
pozostają osobnym składnikiem nagrody i podstawą ocen. Nie ma PER, n-step ani dueling.
Użyj nowego katalogu wyników; parametry i stan bonusu są zachowywane przy wznowieniu.
Detektor obserwuje natywne klatki RGB; CNN nadal dostaje cztery obrazy 84×84.
Licznik bonusu zależy od historii wymiany i nie jest dodatkowym wejściem CNN.
To eksperymentalna zmiana nagród; poprawę oceniamy po prawdziwych punktach.
