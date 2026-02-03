# Electronic Discovery (Mono-GraphMD) for Corona induced RIEF/AN Modeling

This repository implements a **Graph-based model discovery (GraphMD)** framework for discovering compact empirical laws for **corona-induced Audible Noise (AN)** and **Radio Interference / RIEF** in high-voltage transmission lines.
The method uses a **genetic algorithm** to search the symbolic expression space and automatically identifies formulas that both fit experimental data and satisfy **monotonicity constraints**.

---

## 1. Repository Contents

The `uploaded version/` folder contains:

* `Discovery_of_corona.py` — main entry script (training / validation / replay)
* `GSR_multi.py` — core algorithm (graph generation, graph→expression, genetic evolution, fitness evaluation)
* `utils.py` — utilities (linear-coefficient regression, prediction-function construction, monotonicity grid, etc.)
* `dataset/combined_data_AN.xlsx` — AN dataset
* `dataset/combined_data_RIEF.xlsx` — RIEF dataset
* `result_save/AN_3_mono/*` — saved results for the AN task
* `result_save/RIEF_4_mono/*` — saved results for the RIEF task

---

## 2. Method Overview

We encode candidate formulas as **computational graphs**:

* **Nodes**: operators (`add`, `mul`, `log`, `exp`), variables, constants, etc.
* **Edge attributes**: coefficients, exponents, and other operator-coupled parameters.

### Variable convention (as used in scripts)

* `x1`: electric field **E**
* `x2`: spacing **d**
* `x3`: **N**
* `x4`: **I** (present in the dataset; often treated as a fixed input)

### Algorithm workflow

1. Randomly generate an initial population of expression graphs
2. Convert each graph to a SymPy expression
3. Fit linear coefficients by least squares (given the symbolic structure)
4. Define fitness as `(1 - R²) + monotonicity penalty`
5. Evolve graphs with genetic operators (crossover / mutation / elitism / diversity injection)
6. Save `best_graphs.pkl` and `best_fitness.pkl` every 10 generations

---

## 3. Data Format

In `Discovery_of_corona.py`, `process_data()` expects Excel columns named like:

* `X_d{...}_N{...}_I{...}`
* `Y_d{...}_N{...}_I{...}`

Each group is parsed into tuples of the form `[X, d, N, I, y]` and then aggregated into the training matrix.

---

## 4. How to Run

> The current scripts are controlled by editing variables in the source code (no CLI arguments are provided).

In `Discovery_of_corona.py`, set:

* `MODE = 'Train'` — train and save results
* `MODE = 'Valid'` — validate built-in candidate expressions and compute errors
* `MODE = 'Valid_discovered'` — load `result_save/*` and replay historical best expressions
* `Target = 'AN'` or `Target = 'RIEF'`

Then run:

```bash
python "uploaded version/Discovery_of_corona.py"
```

---

## 5. Dependencies

Below is the **package list implied by the project imports**.
Because exact version numbers depend on your local environment, the most reliable way to obtain **the precise versions you are using** is to generate them automatically (see next subsection).

### 5.1 Required Python packages

* `numpy`
* `pandas`
* `scipy`
* `sympy`
* `scikit-learn`
* `matplotlib`
* `networkx`
* `tqdm`

### 5.2 ML / graph-related packages

* `torch`
* `torch-geometric` (imported as `torch_geometric`)

### 5.3 Standard library / runtime utilities

* `os`, `math`, `random`, `time`, `heapq`, `pickle`, `warnings`, `copy`
* `concurrent.futures` (for `ProcessPoolExecutor`)

## 6. Notes

* Expressions involving `log`, division, or roots may require domain checks (e.g., positive inputs) during evaluation.
* The monotonicity penalty is applied on a predefined grid (see `utils.py`), which you can adjust to match your physical priors.

