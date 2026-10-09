# MCMC Can See Can't See - Code Release

Code for *MCMC Can See Can't See*.

## Abstract

It is folklore that passing "convergence" diagnostics does not guarantee that a Markov
chain has converged. We prove this as an impossibility for any finite checklist of
diagnostics. Running more chains, longer chains, and smarter diagnostics does not avoid
this impossibility in certifying convergence. Under regularity assumptions, we determine
how the worst-case undetected non-convergence decays as more diagnostics are added, and
how to choose diagnostics optimally. Folklore suggests that the sampler's slowest mixing
modes are the "dangerous" ones that must be monitored; we prove that the slow modes
yield nearly optimal diagnostics as sampling proceeds. Given a source class, we also
give an anytime-valid bound on the remaining discrepancy and show how undetected bias
caps the effective sample size that a finite checklist can certify.

## Setup

```bash
uv sync --locked
```

Python >=3.11 required; pinned dependencies in `uv.lock`. Datasets and the results
used to plot the paper are included. Install the `sampling` extra for NUTS references:

```bash
uv sync --locked --extra sampling
```

## Reproducing figures

Run from the repository root:

```bash
uv run python scripts/figures/fig_intro_trapped_mixture.py
uv run python scripts/figures/paper_main_figures.py
uv run python scripts/figures/fig_b_learned_alignment.py
uv run python scripts/figures/fig_r3b_suite.py \
    --compact-json paper/figures/data/r3b_suite/main-v2/r3b_main_compact.json \
    --out paper/figures/fig_r3b_main
```

These read the saved results and write all five figures as PDF and PNG to
`paper/figures/`. They run on CPU. The second command produces Figure 2 and the
residual appendix figure. Numerical results are in `paper/figures/data/`.

## Running experiments

Set the output directory, then run a small circle experiment:

```bash
mkdir -p output/paper_experiments
export MCMC_PAPER_OUT="$PWD/output/paper_experiments"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run python -m mcmc.experiments.exp_r1_residual_vs_diagnostics \
    phase=smoke run_id=circle-smoke n_workers=1
```

Experiment modules are in `src/mcmc/experiments/`, with matching YAML files in
`configs/experiment/`. Use `phase=main` for the paper's sampling budgets and a new
`run_id` for each run. `--cfg job --resolve` prints a module's settings; command-line
arguments override them. The synthetic anytime config also has phase and target
overrides, which take precedence over command-line values.

| Paper result | Runner |
|---|---|
| Figure 1: trapped mixture | `scripts/figures/run_intro_mixture.py` |
| Figure 2A: fixed circle and regression laws | `scripts/paper_experiments/a_circle_fixed_laws.py`, `a_dense_regression.py` |
| Figure 3: circle attainment | `scripts/paper_experiments/a_polynomial_control.py`; `exp_r1_residual_vs_diagnostics` |
| Figure 3: synthetic constructions and projected residuals | `exp_r1b_constructions_suite` (`part=constructions` or `projected`); `exp_r1b_surrogate_check` |
| Figure 3: regression residuals | `exp_r1c_real` |
| Figures 2B and 4: regression alignment | `exp_b_real_alignment` |
| Figure 2C: common-horizon regression bounds | `exp_c_common_horizon` |
| Figure 5 and synthetic sensitivity appendix | `exp_r3b_anytime_suite` |
| Appendix: circle and regression initialization comparisons | `exp_r3_anytime_usefulness`, `exp_r3_concrete`, `exp_r3_regression` |
| Appendix: logistic alignment | `exp_b_logistic`, using `exp_r2_german_credit` outputs |
| Appendix: logistic stopping and replication | `scripts/paper_experiments/c_realdata_budget_followup.py`, `c_realdata_replication.py` |

Names starting with `exp_` run as `uv run python -m mcmc.experiments.NAME`.
The `exp_r1b_gapfill*.yaml` configs cover the residual sensitivity experiments;
`exp_r3_anytime_usefulness_stress_fastinit.yaml` is the post hoc circle comparison.
Select these with `--config-name NAME` (without `.yaml`).

The analytic calculations can be run directly:

```bash
uv run python scripts/paper_experiments/a_circle_fixed_laws.py --output output/a-circle
uv run python scripts/paper_experiments/a_polynomial_control.py --output output/a-attainment
uv run python scripts/paper_experiments/a_dense_regression.py --out output/a-regression
```

For the five-dataset common-horizon experiment:

```bash
for dataset in concrete energy airfoil wine_red kin8nm; do
    uv run python -m mcmc.experiments.exp_c_common_horizon \
        dataset="$dataset" phase=main run_id="${dataset}_main" device=cuda
done
uv run python scripts/paper_experiments/c_common_horizon_export.py --phase main
```

Regression alignment uses `phase=pilot` with `part=setup`, then `population`, then
`learned`, passing the resulting directories as `setup_run` and `population_run`.
The included
[selection.json](paper/figures/data/r2_real_alignment/selection.json) contains the
independently selected diagnostic counts; `part=precision` and `learned_refine` apply its precision refinements.

Synthetic anytime runs need an independent `mode=reference` run before
`mode=ensembles`, with its ID passed as `reference_run_id`. The paper uses two
500-ensemble runs per target/sampler, with `rep_offset=0` and `500`. The primary
targets are `gauss-d10-c25`, `mixture-d10-s6-w0.5`, `studentt-d10-nu3` and
`funnel-d10-sy3`, each with `__rwm` and `__mala`. Sensitivity settings are in
[overview.csv](paper/figures/data/r3b_suite/stress-v2/overview.csv). Analysis uses
`scripts/paper_experiments/analyze_r3b.py`, which expects run IDs of the form
`TARGET__SAMPLER__rbf-ell__eps0.8__main-v2` and `__main-ext-v2` for the second chunk.
`extract_r3b_paper_data.py` converts its output to compact plotting data.

Logistic experiments first need NUTS reference draws, which are too large to include:

```bash
uv run python -m mcmc.paperexp.german_credit_reference \
    mode=run 'sets=[hier_c_dense,hier_nc_diag]' role=reference
uv run python -m mcmc.paperexp.german_credit_reference \
    mode=run 'sets=[hier_c_dense]' role=kernel_calibration
```

The reference YAML has presets for all six models. Evaluation, starting-state and
calibration sets have separate roles in `exp_r2_german_credit.yaml`. Generate the
chain outputs with `stage=gpu`, then `cpu`, then `reeval`. Logistic alignment uses
`exp_b_logistic` stages `calibrate`, `select`, `amend_select`, `fresh_gpu`,
`fresh_cpu`, `expanded_gpu`, `expanded_cpu`, `publish2`; its config specifies the
additional NUTS pools for the expanded comparison. The stopping and replication
scripts list their stages with `--help`.

Logistic results use approximate NUTS references and assumed source radii. They
are empirical comparisons. Synthetic coverage is conditional on the independent
reference intervals and the quadrature error allowances recorded with the results.

## Compute

Experiments used four RTX 4090s and CPUs. The five-dataset common-horizon study took
15.73 GPU-hours. Full runs need Linux; PyTorch uses CUDA and NUTS runs use CPU JAX.
Seeds and sampling budgets are set in the experiment configs.

## Tests

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TORCHINDUCTOR_CACHE_DIR="$PWD/output/torchinductor" \
TRITON_CACHE_DIR="$PWD/output/triton" \
uv run --locked --extra sampling pytest
```

The suite takes about 25 minutes. Set `GC_TEST_CUDA=1 MCMC_TEST_CUDA=1` to include
GPU tests. The cache paths allow compiled kernels when `/tmp` is mounted `noexec`.

## Data and license

Code: MIT (see [LICENSE](LICENSE)). Raw datasets and source records are
in `data/raw/`; loaders enforce pinned hashes before preprocessing.

| Dataset | Attribution and source | License |
|---|---|---|
| Concrete Compressive Strength | [I.-C. Yeh (1998)](https://doi.org/10.24432/C5PK67) | CC BY 4.0 |
| Energy Efficiency | [A. Tsanas and A. Xifara (2012)](https://doi.org/10.24432/C51307) | CC BY 4.0 |
| Airfoil Self-Noise | [T. Brooks, D. Pope and M. Marcolini (1989)](https://doi.org/10.24432/C5VW2C) | CC BY 4.0 |
| Wine Quality (red) | [P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis (2009)](https://doi.org/10.24432/C56S3T) | CC BY 4.0 |
| kin8nm | [Z. Ghahramani / DELVE](https://www.cs.toronto.edu/~delve/data/kin/desc.html), [OpenML 189](https://www.openml.org/api/v1/json/data/189) | OpenML: Public |
| German Credit | [H. Hofmann (1994)](https://doi.org/10.24432/C5NC77) | CC BY 4.0 |
| Pima Indians Diabetes | [J. W. Smith et al. (1988)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/), [OpenML 37](https://www.openml.org/api/v1/json/data/37) | OpenML: Public |
| Australian Credit Approval | [R. Quinlan (1987)](https://doi.org/10.24432/C59012) | CC BY 4.0 |
| Ionosphere | [V. Sigillito, S. Wing, L. Hutton and K. Baker (1989)](https://doi.org/10.24432/C5W01B) | CC BY 4.0 |
| Sonar | [T. Sejnowski and R. Gorman (1988)](https://doi.org/10.24432/C5T01Q) | CC BY 4.0 |

UCI datasets use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
OpenML labels kin8nm and Pima as Public without naming a standard license.
Regression predictors and responses use sample standard deviations; logistic
predictors use population standard deviations. Energy uses heating load,
Ionosphere drops its constant column, and Pima's recorded zeros are retained.
The full datasets define posterior targets; there is no train/test split.
