# Regression datasets

Downloaded 2026-10-05 (UTC).

| dir | dataset | source | files |
|---|---|---|---|
| energy | UCI 242 Energy efficiency (768 x 8; response Y1 heating load; Y2 dropped) | https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip ; CSV export https://archive.ics.uci.edu/static/public/242/data.csv | energy_efficiency.zip (ENB2012_data.xlsx, primary), energy_data_uci.csv (cross-check) |
| airfoil | UCI 291 Airfoil self-noise (1503 x 5) | https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip | airfoil_self_noise.zip |
| wine_red | UCI 186 Wine quality, red file (1599 x 11; response quality) | https://archive.ics.uci.edu/static/public/186/wine+quality.zip | wine_quality.zip |
| kin8nm | OpenML 189 kin8nm (DELVE; 8192 x 8; response y) | https://openml.org/data/v1/download/3626/kin8nm.arff (OpenML md5 b36414070701ab8bc985104c3ebd9b77 verified) | kin8nm.arff |

The loaders verify pinned SHA256 hashes before parsing.
Cite the UCI / OpenML records and original authors when reusing the data.
