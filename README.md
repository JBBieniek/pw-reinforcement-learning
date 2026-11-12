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
DQN i DDQN porównuj przy tych samych seedach, ustawieniach gry i liczbie decyzji,
w osobnych katalogach. Jeden krok to domyślnie około czterech klatek emulatora.
Nowe treningi używają LeakyReLU i deterministycznych operacji; obie metody muszą
mieć te same ustawienia. Stare modele zachowują ReLU. Kopiuj wagi razem z `.config.json`.
`model.health.json`/`.jsonl` pokazują cechy obrazu i gradienty; wykryte zapadnięcie
zatrzymuje trening z zapisem diagnostycznym. Krótki test nie gwarantuje wyniku po 15 mln.
