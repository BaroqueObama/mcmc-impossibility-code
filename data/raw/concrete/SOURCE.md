# UCI Concrete Compressive Strength (id 165)

Downloaded 2026-10-05 (UTC) with curl:

- `concrete_compressive_strength.zip` <- https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip
  (contains `Concrete_Data.xls`, SHA256 710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4,
  the full-precision source, and `Concrete_Readme.txt`)
- `concrete_data_uci.csv` <- https://archive.ics.uci.edu/static/public/165/data.csv (the UCI API `data_url`;
  predictors printed to 1 decimal, response to 2 decimals; used only to cross-check the XLS parse)

The loader verifies pinned SHA256 hashes. Reuse requires retaining the copyright notice of
Prof. I-Cheng Yeh and citing: I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial
neural networks," Cement and Concrete Research, 28(12):1797-1808, 1998. Dataset DOI 10.24432/C5PK67.
