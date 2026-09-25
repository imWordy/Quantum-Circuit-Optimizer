# GNN-Guided Iterative Optimization of Small Quantum Circuits

Research prototype combining procedural candidate generation, exact quantum equivalence verification, and Graph Neural Networks (GNN) for structure-aware quantum circuit optimization.

## Project Structure

```
├── config/
│   └── config.yaml             # Central configuration (qubits, gates, seeds, tolerances)
├── src/
│   └── qco/                    # Core Python package
│       ├── circuits/           # Canonical circuit models, serialization, and procedural generator
│       ├── transformations/    # Rule definitions and candidate transformations
│       ├── verification/       # Independent quantum simulation & equivalence checks
│       ├── metrics/            # Circuit cost metrics (gate counts, depth, reductions)
│       ├── graphs/             # Circuit-to-graph translation
│       ├── data/               # Schemas and dataset builders
│       ├── models/             # GNN and baseline models
│       ├── optimization/       # Search and optimization algorithms
│       ├── experiments/        # Experiment execution harnesses
│       ├── evaluation/         # Metrics and report generation
│       └── utils/              # Configuration, reproducibility, logging helpers
├── tests/                      # Automated pytest test suites
├── data/                       # Datasets (raw, processed, splits)
├── models/                     # Saved checkpoints
├── experiments/                # Experiment configurations and outputs
└── results/                    # Output logs, tables, plots
```

## Setup & Installation

1. Create and activate a Python 3.10–3.12 virtual environment:
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. Install dependencies and the `qco` package in editable mode:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

## Running Tests

Run the complete test suite across all implemented phases:
```bash
pytest -v tests/
```

---

## Phase Execution Progress

| Phase | Description | Status | Key Components & Artifacts |
|---|---|---|---|
| **Phase 1** | **Environment and Project Skeleton** | ✅ Complete | Repo skeleton, Python 3.12 venv, dependencies, `config.yaml`, `set_seed()`, `setup_logger()`, `test_environment.py` |
| **Phase 2** | **Canonical Quantum Circuit Representation** | ✅ Complete | `GateOperation`, `CircuitData`, SHA-256 circuit IDs, Qiskit bidirectional conversion, JSON serialization, `test_circuits.py` |
| **Phase 3** | **Small Circuit Generator** | ✅ Complete | `CircuitGenerator`, parameterized angle sampling, diverse batch generation, distribution logging, `test_generator.py` |
| **Phase 4** | **Circuit Cost Metrics** | ✅ Complete | `CircuitCost`, critical path DAG-based `calculate_depth()`, `compute_cost_reduction()`, `test_metrics.py` |
| **Phase 5** | Exact Quantum Equivalence Verification | ⏳ Planned | Unitary comparison & statevector simulation with global phase handling |
| **Phase 6** | Valid Transformation Library | ⏳ Planned | Deterministic quantum algebraic identity rewrite rules |
| **Phase 7** | Procedural Candidate Generator | ⏳ Planned | Transformation application & metadata tracking |
| **Phase 8** | Candidate Verification & Ground-Truth Dataset | ⏳ Planned | Verified dataset generation pipeline |
| **Phase 9** | Dataset Splitting & Leakage Prevention | ⏳ Planned | Grouped split by parent circuit ID |
| **Phase 10** | Circuit Graph Representation | ⏳ Planned | Structural PyG graph encoding preserving gate dependencies |
| **Phase 11** | Non-GNN Baselines | ⏳ Planned | Random search and greedy optimization baselines |
| **Phase 12** | GNN Dataset Interface | ⏳ Planned | PyG `InMemoryDataset` with graph pairs and candidate features |
| **Phase 13** | Initial GNN Model | ⏳ Planned | GNN encoder with prediction head for gate reduction |
| **Phase 14** | GNN Training | ⏳ Planned | Training loop, validation, metrics (MSE, MAE, R²) |
| **Phase 15** | GNN Candidate Ranking | ⏳ Planned | Candidate prioritization and rank correlation evaluation |
| **Phase 16** | GNN-Guided Iterative Optimizer | ⏳ Planned | End-to-end optimizer with Qiskit equivalence verification |
| **Phase 17** | Feedback Learning | ⏳ Planned | Online observation collection and model fine-tuning |
| **Phase 18** | Final Experimental Framework | ⏳ Planned | Comparative benchmark on held-out test suite |
| **Phase 19** | Ablation Studies | ⏳ Planned | Graph structure vs aggregate features ablation |
| **Phase 20** | Final Validation & Reproducibility | ⏳ Planned | End-to-end automated reproducibility check |
| **Phase 21** | Final Demonstration | ⏳ Planned | End-to-end circuit optimization showcase |

---

## Quick Usage Examples

### 1. Generating & Serializing Circuits
```python
from qco.circuits import CircuitGenerator, serialize_to_json, deserialize_from_json

# Initialize generator with configured limits
generator = CircuitGenerator.from_config()
circuit = generator.generate_circuit()

print(f"Generated circuit {circuit.circuit_id} with {circuit.num_qubits} qubits and {len(circuit)} gates.")

# Serialize to JSON
json_str = serialize_to_json(circuit)
recovered = deserialize_from_json(json_str)
assert circuit == recovered
```

### 2. Evaluating Circuit Cost and Reduction
```python
from qco.metrics import compute_circuit_cost, compute_cost_reduction

cost = compute_circuit_cost(circuit)
print(f"Total gates: {cost.total_gates}, CNOTs: {cost.cnot_count}, Depth: {cost.depth}")

# Evaluate reduction against a candidate circuit
reduction = compute_cost_reduction(original=circuit, candidate=recovered)
print(f"Gate reduction: {reduction.gate_reduction} ({reduction.gate_reduction_pct:.1f}%)")
```
