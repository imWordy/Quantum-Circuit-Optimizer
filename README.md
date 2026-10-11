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
│       ├── data/               # Schemas, dataset builder, splitting, candidate features & GNNDataset
│       ├── graphs/             # Circuit-to-graph translation (PyTorch Geometric)
│       ├── models/             # Baseline models, GNN architectures, GNNTrainer, Ranker & FeedbackLoopManager
│       ├── optimization/       # Optimizer suite (RandomOptimizer, GreedyOptimizer, GNNGuidedOptimizer)
│       ├── experiments/        # Experiment execution harnesses
│       ├── evaluation/         # Regression & ranking metrics (MAE, RMSE, R², Spearman rho, top-k)
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
| **Phase 11** | **Non-GNN Baselines** | ✅ Complete | `BaseOptimizer`, `RandomOptimizer` (Baseline A), `GreedyOptimizer` (Baseline B), `AggregateFeatureMLP` (Baseline C), `test_baselines.py` |
| **Phase 12** | **GNN Dataset Interface** | ✅ Complete | `CircuitCandidateData`, `GNNDataset`, `extract_candidate_features()`, zero test-label leakage, PyG batching, `test_gnn_dataset.py` |
| **Phase 13** | **Initial GNN Model** | ✅ Complete | `CircuitGNN` (GCN/GAT layers, residual connections, global mean/max pooling, candidate fusion head), regression training, `test_gnn_model.py` |
| **Phase 14** | **GNN Training** | ✅ Complete | `GNNTrainer`, training/validation loop, MAE/RMSE/R² metrics, model checkpointing, evaluation against trivial mean predictor, `test_trainer.py` |
| **Phase 15** | **GNN Candidate Ranking** | ✅ Complete | `CandidateRanker`, `RankedCandidate`, `compute_ranking_metrics` (Spearman rho, top-1/top-k success), independent verification rule enforcement, `test_ranker.py` |
| **Phase 16** | **GNN-Guided Iterative Optimizer** | ✅ Complete | `GNNGuidedOptimizer`, end-to-end loop: generate → GNN rank → verify equivalence → strictly beneficial gate reduction acceptance, cycle prevention, `test_gnn_optimizer.py` |
| **Phase 17** | **Feedback Learning** | ✅ Complete | `FeedbackRecord`, `FeedbackLoopManager`, online observation logging during optimization, incremental model retraining rounds without divergence, `test_feedback.py` |
| **Phase 18** | **Final Experimental Framework** | ✅ Complete | `BenchmarkRecord`, `BenchmarkHarness`, fair multi-method comparison (Random, Greedy, GNN, MLP), all 16 metrics logged, `test_benchmark.py` |
| **Phase 19** | **Ablation Studies** | ✅ Complete | `AblationFramework`, `AblationStudyResult`, structural graph vs aggregate features (A/B/C/D), rule subset ablations, `test_ablation.py` |
| **Phase 20** | **Final Validation & Reproducibility** | ✅ Complete | `ReproducibilityRunner`, `ReproducibilityReport`, automated end-to-end run, exact circuit reconstruction, metric re-computation, `test_reproducibility.py` |
| **Phase 21** | **Final Demonstration** | ✅ Complete | `FinalDemonstrator`, `DemonstrationReport`, full 5-stage pipeline trace on unseen circuits, formatted comparison report, `test_demonstration.py` |

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

### 7. Optimizing Circuits with Baselines (Random & Greedy)
```python
from qco.circuits import CircuitData, GateOperation
from qco.optimization import GreedyOptimizer, RandomOptimizer

circuit = CircuitData(
    num_qubits=2,
    gates=[
        GateOperation("H", (0,)),
        GateOperation("H", (0,)),
        GateOperation("CX", (0, 1)),
        GateOperation("CX", (0, 1)),
        GateOperation("X", (0,)),
    ],
)

greedy = GreedyOptimizer(max_iterations=10)
result = greedy.optimize(circuit)

print(f"Initial gates: {result.initial_cost.total_gates} -> Final gates: {result.final_cost.total_gates}")
print(f"Gate reduction: {result.gate_reduction}, Equivalent: {result.is_equivalent}")
print(f"Evaluations: {result.candidate_evaluations}, Equivalence checks: {result.equivalence_checks}")
```

### 8. End-to-End Optimization with GNN-Guided Iterative Optimizer
```python
from qco.models import CircuitGNN
from qco.optimization import GNNGuidedOptimizer

# Initialize trained GNN model and guided optimizer
model = CircuitGNN(hidden_dim=64, gnn_type="gcn")
optimizer = GNNGuidedOptimizer(model=model, max_iterations=20)

# Run optimization: candidates ranked by GNN -> verified by Phase 5 -> accepted monotonically
res = optimizer.optimize(circuit)

print(f"Optimizer: {res.optimizer_name}")
print(f"Reduction: {res.gate_reduction} gates (from {res.initial_cost.total_gates} down to {res.final_cost.total_gates})")
print(f"Equivalent: {res.is_equivalent}, Evaluations: {res.candidate_evaluations}, Checks: {res.equivalence_checks}")
```

### 9. Feedback Learning with Online Observation Updates
```python
from qco.models import FeedbackLoopManager

# Wrap model and dataset splits in the feedback manager
manager = FeedbackLoopManager(model=model, initial_splits=splits, lr=0.001)

# Attach manager as feedback collector to capture real optimization evaluations
optimizer = GNNGuidedOptimizer(model=model, feedback_collector=manager)
optimizer.optimize(circuit)

# Step feedback round: incorporates newly verified records and fine-tunes GNN
round_res = manager.step_feedback_round(new_records=new_verified_records, new_circuits=new_circuit_dict)
print(f"Feedback Round {round_res.round_idx}: Val MAE improved from {round_res.val_mae_before:.4f} to {round_res.val_mae_after:.4f}")
```

### 10. Fair Comparative Benchmark Across All Optimizers
```python
from qco.experiments import BenchmarkHarness
from qco.models import CircuitGNN
from qco.models.baselines import AggregateFeatureMLP

# Setup harness with test circuits and trained models
gnn_model = CircuitGNN(hidden_dim=64, gnn_type="gcn")
mlp_model = AggregateFeatureMLP(hidden_dim=32)

harness = BenchmarkHarness(
    test_circuits=[circuit1, circuit2],
    gnn_model=gnn_model,
    mlp_model=mlp_model,
    max_iterations=15,
)

records = harness.run_benchmark(test_circuits)
summary = harness.compute_summary_table()
print(summary[["method", "avg_gate_reduction", "avg_evaluations", "avg_runtime_sec"]])
```

### 11. Component Ablation Studies (Graph Structure vs. Aggregate Features)
```python
from qco.experiments import AblationFramework
from qco.models import CircuitGNN
from qco.models.mlp_baseline import AggregateFeatureMLP

# Setup ablation framework across test circuits
ablation = AblationFramework(
    test_circuits=held_out_circuits,
    gnn_model=gnn_model,
    mlp_model=mlp_model,
    max_iterations=15,
)

# 1. Structural graph vs Aggregate features (A: Random, B: Greedy, C: MLP, D: GNN)
structural_study = ablation.run_structural_vs_aggregate_ablation()
print(structural_study.summary_df)

# 2. Transformation rule subset ablations (e.g. cancellation vs rotation vs all)
rule_study = ablation.run_rule_subset_ablation()
ablation.save_ablation_results(structural_study, output_dir="results/ablation")
```

### 12. Automated End-to-End Reproducibility Pipeline
```python
from qco.experiments import ReproducibilityRunner

# Initialize runner with path to output results directory
runner = ReproducibilityRunner(output_dir="results/reproducibility", seed=42)

# Executes complete automated pipeline:
# 1. Environment metadata capture (software versions, OS, git commit)
# 2. Procedural circuit generation & verified ground-truth dataset creation
# 3. Disjoint dataset splitting with zero data leakage
# 4. GNN model training and validation loss tracking
# 5. Held-out circuit optimization with GNN and Greedy optimizers
# 6. Strict post-optimization correctness checks:
#    - Circuit reconstruction (original & optimized)
#    - Independent exact unitary equivalence check (tolerance <= 1e-7)
#    - Metric recalculation and consistency check
report = runner.run_full_pipeline(num_circuits=10, training_epochs=5, max_optimization_iters=15)
print(f"Reproducibility verification passed: {report.overall_success}")
```

### 13. End-to-End Final Demonstration on Unseen Circuits
```python
from qco.circuits import CircuitData, GateOperation
from qco.experiments import FinalDemonstrator
from qco.models import CircuitGNN

# Define or sample an unseen circuit
unseen_circuit = CircuitData(
    num_qubits=2,
    gates=[
        GateOperation("CX", (0, 1)),
        GateOperation("CX", (0, 1)),
        GateOperation("RZ", (0,), params=[0.5]),
        GateOperation("RZ", (0,), params=[-0.5]),
        GateOperation("H", (1,)),
        GateOperation("H", (1,)),
        GateOperation("X", (0,)),
    ],
)

# Run complete 5-stage decision pipeline (Generation -> Ranking -> Verification -> Cost -> Acceptance)
demonstrator = FinalDemonstrator(model=gnn_model, max_iterations=20)
report = demonstrator.run_demonstration(unseen_circuit, verbose=True)

# Prints formatted comparative performance report
print(report.format_summary_table())
demonstrator.save_demonstration(report, output_dir="results/demonstration")
```



