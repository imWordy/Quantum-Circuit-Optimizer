# GNN-Guided Iterative Optimization of Small Quantum Circuits

Research prototype combining procedural candidate generation, exact quantum equivalence verification, and Graph Neural Networks (GNN) for structure-aware quantum circuit optimization.

## Project Structure

```
├── config/
│   └── config.yaml             # Central configuration (qubits, gates, seeds, tolerances)
├── src/
│   └── qco/                    # Core Python package
│       ├── circuits/           # Canonical circuit models, serialization, and procedural generator
│       ├── transformations/    # Valid rewrite rules, candidate generator, and rule validator
│       ├── verification/       # Independent quantum simulation & equivalence checks
│       ├── metrics/            # Circuit cost metrics (gate counts, depth, reductions)
│       ├── data/               # Schemas, dataset builder, and leakage-free dataset splitting
│       ├── graphs/             # Circuit-to-graph translation (PyTorch Geometric)
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
| **Phase 5** | **Exact Quantum Equivalence Verification** | ✅ Complete | Independent correctness authority: `compute_unitary()`, `check_equivalence()`, global phase handling, `test_verification.py` |
| **Phase 6** | **Valid Transformation Library** | ✅ Complete | `SelfInverseCancellation`, `CNOTCancellation`, `RotationCombination`, `ZeroRotationRemoval`, `IdentityRemoval`, `RuleValidator`, `test_transformations.py` |
| **Phase 7** | **Procedural Candidate Generator** | ✅ Complete | `Candidate`, `CandidateGenerator` with full metadata (parent ID, rule name, indices, preliminary cost, verification status), `test_candidate_generator.py` |
| **Phase 8** | **Candidate Verification & Ground-Truth Dataset** | ✅ Complete | `ObservationRecord`, `DatasetBuilder` (full pipeline: generate → verify → cost → store), JSONL/CSV/circuit save-load, `test_dataset_builder.py` |
| **Phase 9** | **Dataset Splitting & Leakage Prevention** | ✅ Complete | `DatasetSplits`, grouped disjoint splitting by `parent_circuit_id`, `validate_leakage()`, save/load split partitions, `test_dataset_splits.py` |
| **Phase 10** | **Circuit Graph Representation** | ✅ Complete | `circuit_to_graph()`, PyTorch Geometric `Data` encoding, temporal order & qubit dependency edges, structural distinguishability, `test_circuit_graph.py` |
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

generator = CircuitGenerator.from_config()
circuit = generator.generate_circuit()
print(f"Circuit {circuit.circuit_id}: {circuit.num_qubits} qubits, {len(circuit)} gates")

json_str = serialize_to_json(circuit)
recovered = deserialize_from_json(json_str)
assert circuit == recovered
```

### 2. Evaluating Circuit Cost and Reduction
```python
from qco.metrics import compute_circuit_cost, compute_cost_reduction

cost = compute_circuit_cost(circuit)
print(f"Total gates: {cost.total_gates}, CNOTs: {cost.cnot_count}, Depth: {cost.depth}")
```

### 3. Exact Quantum Equivalence Verification
```python
from qco.circuits import CircuitData, GateOperation
from qco.verification import check_equivalence

c1 = CircuitData(num_qubits=1, gates=[GateOperation("H", (0,)), GateOperation("Z", (0,)), GateOperation("H", (0,))])
c2 = CircuitData(num_qubits=1, gates=[GateOperation("X", (0,))])

result = check_equivalence(c1, c2)
print(f"Equivalent: {result.equivalent}, Matrix Distance: {result.matrix_distance:.2e}")
```

### 4. Applying Transformation Rules
```python
from qco.transformations import SelfInverseCancellation, RuleValidator

circuit = CircuitData(num_qubits=1, gates=[GateOperation("H", (0,)), GateOperation("H", (0,))])
rule = SelfInverseCancellation()
validator = RuleValidator()

for match in rule.find_matches(circuit):
    result = validator.validate_transformation(rule, circuit, match)
    print(f"Rule: {result.rule_name}, Valid: {result.valid}, Gate Δ: {result.actual_gate_reduction}")
```

### 5. Building the Ground-Truth Dataset and Splitting with Leakage Prevention
```python
from qco.circuits import CircuitGenerator
from qco.data import DatasetBuilder, split_dataset_from_config

# Generate circuits and build verified dataset
generator = CircuitGenerator.from_config()
circuits = generator.generate_batch(50)

builder = DatasetBuilder()
builder.process_batch(circuits)

# Split by parent circuit ID (70% train, 15% val, 15% test) to prevent data leakage
splits = split_dataset_from_config(builder.records, circuit_store=builder.circuit_store)
splits.validate_leakage()
splits.save("data/splits/split_v1")

print(f"Train records: {len(splits.train_records)} ({len(splits.train_parent_ids)} parent circuits)")
print(f"Val records:   {len(splits.val_records)} ({len(splits.val_parent_ids)} parent circuits)")
print(f"Test records:  {len(splits.test_records)} ({len(splits.test_parent_ids)} parent circuits)")
```

### 6. Converting Quantum Circuits to Graph Representations (PyG)
```python
from qco.circuits import CircuitData, GateOperation
from qco.graphs import circuit_to_graph

circuit = CircuitData(
    num_qubits=2,
    gates=[
        GateOperation("H", (0,)),
        GateOperation("CX", (0, 1)),
        GateOperation("RX", (0,), (0.5,)),
    ],
)

# Convert to PyTorch Geometric Data object
graph = circuit_to_graph(circuit)
print(f"Graph nodes: {graph.num_nodes}, Node feature shape: {graph.x.shape}")
print(f"Graph edges: {graph.edge_index.shape[1]}, Edge index:\n{graph.edge_index}")
```
