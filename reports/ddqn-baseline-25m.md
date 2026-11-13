# Double DQN — zachowany wynik 25 mln

Trening i wszystkie oceny zakończone. Koniec ostatniej oceny: 13 listopada 2026, 02:03 czasu polskiego.

| Decyzje | Średnia | Mediana | Minimum | Maksimum | Gry ≥864 |
|---:|---:|---:|---:|---:|---:|
| 1,000,000 | 25.87 | 25.00 | 11 | 68 | 0/100 |
| 2,000,000 | 26.40 | 25.50 | 6 | 47 | 0/100 |
| 4,000,000 | 40.90 | 39.00 | 14 | 80 | 0/100 |
| 7,000,000 | 41.24 | 38.50 | 15 | 74 | 0/100 |
| 10,000,000 | 48.51 | 47.50 | 9 | 89 | 0/100 |
| 15,000,000 | 58.10 | 55.50 | 16 | 121 | 0/100 |
| 20,000,000 | 50.49 | 47.00 | 16 | 259 | 0/100 |
| 25,000,000 | 67.30 | 65.50 | 28 | 211 | 0/100 |

Końcowy wynik: średnia 67,30; mediana 65,50; maksimum 211. Najwyższy pojedynczy wynik we wszystkich ocenach: 259 przy 20 mln decyzji. Żaden test nie osiągnął 864.

Protokół: 100 pełnych gier na każdy etap, wspólny zestaw seedów oceny (seed bazowy 3303497698), epsilon oceny 0, automatyczny FIRE na starcie i po utracie życia, limit 27000 decyzji na grę. Żadna gra w oficjalnych ocenach nie osiągnęła limitu.

Środowisko: ALE/Breakout-v5, mode 0, difficulty 0, sticky actions 0,25, 4 klatki emulatora na decyzję. Trening: seed 230849415; CNN z LeakyReLU; deterministyczne operacje; jednostajny replay 50000; epsilon 1→0,1 przez 250000 decyzji, potem 0,1; Adam, lr 0,0001. Surowe nagrody, utrata życia nie kończy celu TD.

Ograniczenia: to jeden seed treningu. Sto gier testowych nie mierzy zmienności między niezależnymi treningami. ALE 0.12.1 nie zapisuje poprzedniej wykonanej sticky action, więc pełny checkpoint zachowuje stan uczenia i gry, ale wznowienie nie gwarantuje trajektorii identycznej z nieprzerwanym treningiem.

Archiwum zawiera pełny checkpoint model.training.pt, ocenione wagi i konfiguracje dla wszystkich ośmiu etapów, surowe wyniki gier, metryki, health checks, logi oraz archiwa źródeł z uruchomienia i przyspieszenia. Nie zawiera dużych pomocniczych checkpointów diagnostycznych.

Oryginalny katalog pozostaje w outputs/ddqn-stable-15m. Zapis repozytorium wskazany przy rozpoczęciu archiwizacji: d7782ffb07330e33e230fa965ded36a2a9073853. Faktyczne źródła historycznych uruchomień dokumentują dołączone archiwa i pliki provenance.json.

manifest.json zawiera SHA-256 plików archiwum i potwierdzenie zgodności każdej kopii z plikiem źródłowym. manifest.json nie zawiera własnego skrótu.
