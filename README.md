# GNN-Guided Iterative Optimization of Small Quantum Circuits

Research prototype combining procedural candidate generation, exact quantum equivalence verification, and Graph Neural Networks (GNN) for structure-aware quantum circuit optimization.

## Project Structure

```
├── config/
│   └── config.yaml             # Central configuration (qubits, gates, seeds, tolerances)
├── src/
│   └── qco/                    # Core Python package
│       ├── circuits/           # Quantum circuit representations and generators
│       ├── transformations/    # Rule definitions and candidate transformations
│       ├── verification/       # Independent quantum simulation & equivalence checks
│       ├── metrics/            # Circuit cost metrics (gate counts, depth)
│       ├── graphs/             # Circuit-to-graph translation
│       ├── data/               # Schemas and dataset builders
│       ├── models/             # GNN and baseline models
│       ├── optimization/       # Search and optimization algorithms
│       ├── experiments/        # Experiment execution harnesses
│       ├── evaluation/         # Metrics and report generation
│       └── utils/              # Configuration, reproducibility, logging helpers
├── tests/                      # Automated test suite
├── data/                       # Datasets (raw, processed, splits)
├── models/                     # Saved checkpoints
├── experiments/                # Experiment configurations and outputs
└── results/                    # Output logs, tables, plots
```

## Setup & Installation

1. Create and activate a Python 3.10-3.12 virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

## Running Tests

Run the full pytest suite:
```bash
pytest tests/
```
