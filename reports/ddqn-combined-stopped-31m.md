# Double DQN combined — zakończony bieg, oceny do 31 mln

Trening zatrzymano 2026-11-14 na prośbę użytkownika przed uruchomieniem rewizji 2.
Ostatni odczyt postępu: **31,359,000 decyzji**. Pełny checkpoint do wznowienia
obejmuje **31 000 000 decyzji**. Późniejsze pliki samych wag nie zawierają replay
ani optymalizatora; dokładny numer decyzji nie jest w nich osadzony.

- Ukończone oceny: **31 × 100 gier**.
- Najlepsza średnia: **178.73** przy **15 mln**; mediana **173.5**, maksimum **358**.
- Ostatnia ocena, **31 mln**: średnia **162.44**, mediana **143**, maksimum **357**.
- Najwyższy pojedynczy wynik: **416** przy **6 mln**.
- Wyniki co najmniej 864: **0 / 3100**. Gry przerwane limitem oceny: **0**.

Archiwum: `outputs/archive/ddqn-combined-stopped-20261114/`.
Wszystkie **164** dostępne pliki skopiowano i porównano z oryginałami przez SHA-256.
Manifest `archive.manifest.json` zawiera hashe i wynik weryfikacji checkpointu.
Wyłączono wyłącznie redundantny folder `preflight-source`, niedostępny wskutek
pierwotnego sandboxowego ACL. Właściwy zamrożony `source/` i `source.zip` zachowano
i zweryfikowano. Nie zmieniano oryginalnych plików ani uprawnień.

Pełny checkpoint zawiera replay 200 tys. przejść i **7,745,001** aktualizacji Adam.
Parametry sieci i stan optymalizatora są skończone. ALE 0.12.1 nie zapisuje ostatniej
zastosowanej akcji sticky, więc wznowienie nie gwarantuje trajektorii identycznej
z treningiem bez przerwy. To jeden seed treningu; wszystkie oceny używają tych
samych 100 seedów testowych. CSV jest niezmienioną kopią `comparison.csv`.
