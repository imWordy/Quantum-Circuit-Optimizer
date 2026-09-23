# GNN-Guided Iterative Optimization of Small Quantum Circuits

## Phase-by-Phase Project Execution Plan

**Purpose:** This document is the implementation contract for the
project. An AI coding agent should treat it as the primary project
specification and follow the phases in order.

------------------------------------------------------------------------

# 0. Project Definition

## 0.1 Project title

**Structure-Aware Deep Learning for Iterative Optimization of Small
Quantum Circuits**

## 0.2 Core objective

Build a research prototype that starts with a small quantum circuit,
procedurally generates valid candidate transformations, uses a Graph
Neural Network (GNN) to rank those candidates, independently verifies
candidate equivalence using quantum simulation, measures circuit-cost
improvements, and feeds verified outcomes back into the learning system.

The core loop is:

``` text
Existing Circuit
      |
      v
Circuit Analysis
      |
      v
Graph Representation
      |
      v
Procedural Candidate Generation
      |
      v
GNN Candidate Ranking
      |
      v
Exact Equivalence Verification
      |
      v
Cost Evaluation
      |
      v
Accept Useful Candidate
      |
      v
Repeat
      |
      v
Verified Result -> Training Data -> GNN Update
```

## 0.3 Separation of responsibilities

- **Procedural system:** generates candidate transformations.
- **GNN:** predicts/ranks which candidates are promising.
- **Qiskit simulator/equivalence checker:** determines whether a
  candidate preserves the computation.
- **Cost evaluator:** determines whether the candidate actually improves
  the circuit.
- **Feedback system:** stores verified observations for future model
  training.

The GNN is never the authority on correctness.

## 0.4 Initial scope

- 2–5 qubits.
- Approximately 5–30 gates.
- Small circuits that can be simulated exactly/classically.
- A controlled gate vocabulary.
- A controlled library of known equivalence-preserving transformations.
- Simulation rather than physical quantum hardware.

Initial optimization metrics:

1.  Total gate count — primary.
2.  CNOT/two-qubit gate count — secondary.
3.  Circuit depth — secondary.

The system must report all three rather than hiding trade-offs behind
one vague “efficiency” score.

## 0.5 Explicitly out of scope

Do not introduce these unless explicitly requested later:

- Reinforcement learning as the main learning method.
- Quantum machine learning.
- Large-scale industrial circuits.
- Physical quantum hardware.
- Automatic invention of new quantum identities.
- Arbitrary circuit generation by a neural decoder.
- Global optimality claims.
- Transformer architectures.
- GANs.
- Variational quantum algorithms as the optimization target.
- Continuous gate-parameter optimization as the primary problem.
- Replacing exact equivalence verification with an ML prediction.

------------------------------------------------------------------------

# 1. Global Rules for the AI Coding Agent

## Rule 1 — Implement phases in order

Do not jump directly to the GNN.

Required progression:

``` text
Environment
-> Circuit Representation
-> Circuit Generation
-> Transformations
-> Verification
-> Cost Evaluation
-> Dataset
-> Graph Representation
-> Baselines
-> GNN
-> GNN Optimization
-> Feedback Learning
-> Experiments
```

## Rule 2 — Every phase must have tests

A phase is complete only when:

- implementation exists;
- tests exist;
- tests pass;
- outputs have been inspected where appropriate;
- results are reproducible.

## Rule 3 — Never fabricate experimental results

Do not hard-code or invent accuracy, gate-reduction percentages,
runtime, model performance, or superiority over baselines.

## Rule 4 — Preserve the original circuit

The canonical circuit representation must always remain recoverable. ML
features and graph representations are derived views, not replacements
for the circuit.

## Rule 5 — Verify independently

The GNN predicts candidate usefulness. Qiskit-based verification
determines equivalence. Never accept a candidate solely because the GNN
predicts a positive result.

## Rule 6 — Keep configuration centralized

Do not scatter qubit limits, gate limits, random seeds, tolerances,
learning rates, or iteration limits throughout the codebase.

## Rule 7 — Prefer simple implementations

This is a research prototype. Do not introduce complex infrastructure
unless required.

------------------------------------------------------------------------

# 2. Target Repository Structure

The agent may adjust filenames if necessary, but responsibilities should
remain separated.

``` text
quantum_circuit_optimizer/
|
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── pyproject.toml
|
├── config/
│   └── config.yaml
|
├── src/
│   └── qco/
│       ├── circuits/
│       │   ├── generator.py
│       │   ├── representation.py
│       │   └── serialization.py
│       ├── transformations/
│       │   ├── rules.py
│       │   ├── candidate_generator.py
│       │   └── validator.py
│       ├── verification/
│       │   ├── simulator.py
│       │   └── equivalence.py
│       ├── metrics/
│       │   └── circuit_cost.py
│       ├── graphs/
│       │   └── circuit_graph.py
│       ├── data/
│       │   ├── schemas.py
│       │   ├── dataset_builder.py
│       │   └── dataset.py
│       ├── models/
│       │   ├── gnn.py
│       │   ├── mlp_baseline.py
│       │   └── training.py
│       ├── optimization/
│       │   ├── random_optimizer.py
│       │   ├── greedy_optimizer.py
│       │   └── gnn_optimizer.py
│       ├── experiments/
│       │   ├── baseline_experiment.py
│       │   ├── gnn_experiment.py
│       │   ├── feedback_experiment.py
│       │   └── ablation_experiment.py
│       └── evaluation/
│           ├── metrics.py
│           └── reports.py
|
├── tests/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── models/
├── experiments/
└── results/
    ├── logs/
    ├── tables/
    └── plots/
```

------------------------------------------------------------------------

# PHASE 1 — Environment and Project Skeleton

## Objective

Create a reproducible Python project with the minimum dependencies and a
clean architecture.

## Required technologies

- Python
- Qiskit
- Qiskit Aer or the appropriate current simulator package
- NumPy
- pandas
- PyTorch
- PyTorch Geometric
- scikit-learn
- SciPy where required
- Matplotlib
- pytest
- Git

Do not add additional ML frameworks unless required.

## Implementation tasks

1.  Create the repository structure.
2.  Create the Python environment.
3.  Add dependency management.
4.  Create the central configuration file.
5.  Create package initialization files.
6.  Create a basic logging system.
7.  Create a reproducibility utility for random seeds.
8.  Create a basic test suite.
9.  Add a README with installation and test instructions.

## Configuration should include at least

``` yaml
seed: 42

circuit:
  min_qubits: 2
  max_qubits: 5
  min_gates: 5
  max_gates: 30

verification:
  tolerance: configurable

optimization:
  max_iterations: configurable

data:
  train_ratio: 0.70
  validation_ratio: 0.15
  test_ratio: 0.15
```

## Testing

Test:

- dependency imports;
- Qiskit circuit creation;
- simulator execution;
- PyTorch tensor creation;
- PyTorch Geometric graph creation;
- seed reproducibility.

### Expected result

A clean test command passes and reports all environment components as
working.

------------------------------------------------------------------------

# PHASE 2 — Canonical Quantum Circuit Representation

## Objective

Define the project’s source-of-truth representation of a quantum
circuit.

It must preserve:

- number of qubits;
- gate order;
- gate type;
- target/control qubits;
- gate parameters;
- enough information to reconstruct the circuit exactly.

Do not use aggregate feature vectors as the canonical representation.

## Initial gate vocabulary

- I
- X
- Y
- Z
- H
- RX
- RY
- RZ
- CX/CNOT

The vocabulary should be centralized and configurable.

## Implementation

Create:

- circuit wrapper/data model;
- circuit ID;
- serialization/deserialization;
- validation;
- conversion to/from Qiskit circuits.

## Testing

### Test A — Round trip

``` text
Circuit
-> serialize
-> deserialize
-> reconstruct
```

Compare the original and reconstructed circuits.

### Test B — Parameter preservation

Use nontrivial RX/RY/RZ parameters.

### Test C — Qubit preservation

Use different control-target assignments.

### Test D — Behavioral preservation

Simulate original and reconstructed circuits.

### Expected result

The reconstructed circuit is structurally and behaviorally equivalent to
the original within configured tolerance.

------------------------------------------------------------------------

# PHASE 3 — Small Circuit Generator

## Objective

Create a procedural generator for valid small quantum circuits.

The generator should vary:

- number of qubits;
- number of gates;
- gate types;
- gate locations;
- rotation parameters.

It creates initial problem instances; it does not optimize them.

## Implementation

1.  Select qubit count within limits.
2.  Select target circuit size.
3.  Sample allowed gates.
4.  Select valid qubits.
5.  Generate valid parameters.
6.  Construct the circuit.
7.  Validate it.
8.  Assign reproducible IDs when seeded.

## Distribution logging

Record:

- number of circuits;
- gate-count distribution;
- depth distribution;
- CNOT distribution;
- qubit distribution;
- gate-type distribution.

## Testing

Generate several hundred circuits and verify:

- every circuit is valid;
- no invalid qubit indices;
- only allowed gates appear;
- parameters are valid;
- all circuits simulate;
- configured limits are respected.

### Expected result

The generator produces diverse valid circuits rather than nearly
identical circuits.

------------------------------------------------------------------------

# PHASE 4 — Circuit Cost Metrics

## Objective

Create deterministic circuit-cost functions.

## Required metrics

- total gate count;
- single-qubit gate count;
- two-qubit gate count;
- CNOT count;
- circuit depth.

## Reduction metrics

For original cost `O` and candidate cost `C`:

``` text
absolute_reduction = O - C
percentage_reduction = (O - C) / O * 100
```

Handle zero denominators explicitly.

## Testing

Create hand-verifiable circuits and compare expected values with
computed values.

Test:

- small circuits;
- single-qubit gates;
- multiple CNOTs;
- parallel gates;
- sequential gates;
- parameterized gates.

### Expected result

All metric calculations match manually calculated values.

------------------------------------------------------------------------

# PHASE 5 — Exact Quantum Equivalence Verification

## Objective

Create the independent correctness authority.

For small circuits, compare their implemented unitary operations where
applicable.

Conceptually:

``` text
Circuit A -> U_A
Circuit B -> U_B
```

The checker must determine equivalence within numerical tolerance and
handle global phase appropriately.

## Implementation

Create:

``` text
check_equivalence(circuit_a, circuit_b)
```

Return structured information such as:

``` text
equivalent
tolerance
comparison_method
global_phase_handling
```

## Testing

Create:

- known equivalent pairs;
- known non-equivalent pairs;
- structurally different but behaviorally equivalent pairs;
- parameterized cases;
- numerical tolerance boundary cases.

### Expected result

Equivalent and non-equivalent circuits are correctly distinguished.

This phase is a hard prerequisite for the transformation pipeline.

------------------------------------------------------------------------

# PHASE 6 — Valid Transformation Library

## Objective

Implement a small, explicit library of mathematically justified
transformations.

The ML model must not invent transformations.

## Initial rules

### Self-inverse cancellation

``` text
X X -> I
Y Y -> I
Z Z -> I
H H -> I
```

### CNOT cancellation

``` text
CX(a,b) CX(a,b) -> I
```

when the required conditions are satisfied.

### Rotation combination

``` text
RX(a) RX(b) -> RX(a+b)
RY(a) RY(b) -> RY(a+b)
RZ(a) RZ(b) -> RZ(a+b)
```

Normalize angles consistently.

### Zero rotation removal

``` text
RX(0) -> I
RY(0) -> I
RZ(0) -> I
```

Do not expand the rule set until the initial pipeline is stable.

## Every rule must define

- rule name;
- mathematical identity;
- applicability conditions;
- transformation function;
- expected effect;
- unit tests.

## Testing

For every rule:

1.  Construct a valid example.
2.  Apply the rule.
3.  Check validity.
4.  Check equivalence.
5.  Check expected cost change.
6.  Test cases where the rule must not apply.

### Expected result

Every implemented transformation passes equivalence and applicability
tests.

------------------------------------------------------------------------

# PHASE 7 — Procedural Candidate Generator

## Objective

Generate possible transformed circuits from a current circuit.

The generator must not decide which candidate is best.

## Process

``` text
Current circuit
     |
Find applicable rule locations
     |
Apply one transformation
     |
Create candidate
     |
Attach metadata
```

## Candidate metadata

Record:

- candidate ID;
- parent circuit ID;
- transformation name;
- transformation location;
- resulting circuit;
- preliminary cost;
- verification status;
- seed if relevant.

A generated candidate is only a proposal until verified.

## Testing

Use a circuit with multiple applicable transformations.

Verify:

- all intended candidates are generated;
- candidates are distinct where expected;
- source circuit is not mutated;
- metadata is correct;
- deterministic generation works with a fixed seed.

### Expected result

Candidate generation is correct, traceable, and side-effect free.

------------------------------------------------------------------------

# PHASE 8 — Candidate Verification and Ground-Truth Dataset

## Objective

Create the complete non-ML data pipeline.

## Pipeline

``` text
Initial circuit
      |
Generate candidates
      |
Verify candidates
      |
Calculate costs
      |
Calculate actual improvements
      |
Store observation
```

## Dataset fields

At minimum:

``` text
parent_circuit_id
candidate_id
transformation_type
transformation_location

parent_gate_count
candidate_gate_count
parent_cnot_count
candidate_cnot_count
parent_depth
candidate_depth

gate_reduction
cnot_reduction
depth_reduction

equivalent
```

Store circuit references rather than duplicating full circuit data where
practical.

## Ground-truth principle

Training targets must come from actual verified outcomes.

Never use future GNN predictions as labels.

## Testing

Randomly inspect records:

1.  Reconstruct parent.
2.  Reconstruct candidate.
3.  Re-run equivalence.
4.  Re-run metrics.
5.  Compare to stored values.

### Expected result

Stored ground truth matches independently recalculated ground truth.

------------------------------------------------------------------------

# PHASE 9 — Dataset Splitting and Leakage Prevention

## Objective

Create scientifically valid train/validation/test splits.

## Critical rule

Split by parent circuit, not blindly by candidate row.

All candidates derived from the same parent circuit must remain in one
split.

Recommended initial split:

``` text
70% training
15% validation
15% test
```

## Testing

Verify that parent IDs are disjoint across all splits.

### Expected result

No parent circuit appears in more than one split.

------------------------------------------------------------------------

# PHASE 10 — Circuit Graph Representation

## Objective

Represent circuits as graphs while preserving useful structural
information.

## Node representation

Each gate becomes a node.

Potential node features:

- gate type;
- layer/depth position;
- qubit information;
- parameter values;
- number of qubits acted upon.

## Edge representation

Edges should encode circuit relationships.

At minimum, operations occurring sequentially on the same qubit should
have dependency/order edges.

Two-qubit operations must preserve their relationship with both qubits.

## Critical requirement

The graph must distinguish circuits that have the same aggregate
statistics but different structure.

For example:

``` text
H -> CX -> RX
```

must remain distinguishable from:

``` text
RX -> CX -> H
```

## Testing

Construct structurally different circuits with identical aggregate
statistics.

Convert them to graphs.

Verify:

- graph structures differ;
- node features are correct;
- qubit relationships are preserved;
- parameters are preserved;
- original circuits are not mutated.

### Expected result

Graph representation preserves meaningful circuit structure.

------------------------------------------------------------------------

# PHASE 11 — Non-GNN Baselines

## Objective

Establish simple optimization baselines before introducing the GNN.

## Baseline A — Random

At each iteration:

1.  Generate candidates.
2.  Select randomly.
3.  Verify.
4.  Accept only according to the defined optimization rule.

## Baseline B — Greedy

At each iteration:

1.  Generate candidates.
2.  Verify.
3.  Calculate cost.
4.  Select the best immediate valid improvement.

## Optional Baseline C — Aggregate-feature MLP

Use:

``` text
qubit count
gate count
CNOT count
depth
single-qubit count
CNOT ratio
```

This supports a later structure-vs-aggregate comparison.

## Testing

Run baselines on the same held-out circuits.

Record:

- initial/final cost;
- gate reduction;
- CNOT reduction;
- depth reduction;
- candidate evaluations;
- equivalence checks;
- iterations;
- runtime.

### Expected result

Both baselines produce valid equivalent outputs.

Do not assume which performs best before experimentation.

------------------------------------------------------------------------

# PHASE 12 — GNN Dataset Interface

## Objective

Prepare graph data for the graph-learning framework.

Each example should provide:

``` text
circuit graph
candidate transformation information
target
```

Initial target:

``` text
gate_reduction
```

This is a regression problem.

## Candidate features

Encode information such as:

- transformation type;
- location;
- affected qubits;
- local context.

Do not leak the future target into input features.

## Testing

Verify:

- graph tensor dimensions;
- candidate-feature dimensions;
- target correctness;
- batching;
- no test-label leakage.

### Expected result

A batch can be passed through a placeholder model without tensor or
batching errors.

------------------------------------------------------------------------

# PHASE 13 — Initial GNN Model

## Objective

Build a small GNN that predicts the benefit of a candidate
transformation.

Initial architecture:

``` text
Circuit Graph
      |
GNN layers
      |
Graph embedding
      |
Candidate features
      |
Concatenate
      |
MLP prediction head
      |
Predicted gate reduction
```

The GNN does not generate circuits.

## Initial task

Regression:

``` text
input  = circuit + candidate
output = predicted gate reduction
```

Start with a standard regression loss such as MSE.

## Testing

Test:

- forward pass;
- graph batching;
- output shape;
- backward pass;
- optimizer step;
- several training iterations.

### Expected result

The model trains without NaNs, tensor-shape errors, or graph-batching
errors.

------------------------------------------------------------------------

# PHASE 14 — GNN Training

## Objective

Train the GNN on verified observations.

Use:

``` text
training split
validation split
test split
```

The test set remains untouched until final evaluation.

## Track

- training loss;
- validation loss;
- epoch;
- model checkpoint;
- MAE;
- RMSE;
- R² where appropriate.

Because the actual application is ranking, ranking metrics should be
added later.

## Testing

Compare against a trivial predictor such as the mean training target.

### Expected result

The GNN should learn measurable signal beyond the trivial predictor.

If it does not, investigate:

- dataset quality;
- target definition;
- graph representation;
- model design;
- leakage;
- transformation distribution.

Do not immediately increase model complexity.

------------------------------------------------------------------------

# PHASE 15 — GNN Candidate Ranking

## Objective

Use the trained GNN to rank candidates.

## Process

``` text
Current circuit
      |
Generate candidates
      |
Convert candidates to graph inputs
      |
GNN predicts benefit
      |
Sort by predicted benefit
      |
Evaluate highest-priority candidates first
```

## Critical rule

Prediction does not equal acceptance.

Every candidate still goes through:

``` text
validity
equivalence
cost evaluation
```

## Testing

For unseen circuits:

1.  Generate candidates.
2.  Record predictions.
3.  Calculate actual outcomes independently.
4.  Compare rankings.

Measure where useful:

- Spearman rank correlation;
- top-1 success;
- top-k success;
- actual improvement among top-k candidates.

------------------------------------------------------------------------

# PHASE 16 — GNN-Guided Iterative Optimizer

## Objective

Integrate the deterministic pipeline and GNN.

## Core loop

``` text
current = original circuit

while stopping condition not met:

    candidates = generate_candidates(current)

    predictions = GNN.rank(candidates)

    ordered_candidates = sort(predictions)

    for candidate in ordered_candidates:

        verify equivalence

        if equivalent:
            calculate cost

            if candidate is beneficial:
                current = candidate
                break

    if no beneficial candidate:
        stop
```

## Acceptance rule

Initially require improvement in the primary metric:

``` text
candidate_gate_count < current_gate_count
```

Secondary metrics are recorded rather than silently overriding the
primary objective.

## Termination

Stop when:

- no beneficial candidate exists;
- maximum iterations reached;
- candidate state already visited;
- configured safety condition triggers.

## Testing

Run on unseen circuits.

Verify:

- every accepted candidate is equivalent;
- primary cost never increases;
- optimizer terminates;
- no repeated state causes an infinite loop;
- final circuit is valid and reconstructable.

------------------------------------------------------------------------

# PHASE 17 — Feedback Learning

## Objective

Use verified optimization outcomes to update the learning system.

Each attempted candidate can produce:

``` text
GNN prediction
+
actual verified outcome
```

Example:

``` text
prediction = 4.2
actual improvement = 2
```

The actual value is ground truth.

## Feedback record

Store:

- model version;
- circuit ID;
- candidate ID;
- predicted benefit;
- actual benefit;
- equivalence result;
- cost metrics;
- optimizer iteration.

## Update strategy

Initial approach:

``` text
collect batch of new observations
      |
append to training data
      |
retrain/update GNN
      |
evaluate
```

Do not add complex online-learning behavior unless explicitly requested.

## Testing

Run multiple feedback rounds.

Compare before/after:

- prediction error;
- ranking quality;
- final optimization result;
- evaluation budget.

### Expected result

The system can incorporate verified new data without breaking.

It is not required that every update improves every metric.

------------------------------------------------------------------------

# PHASE 18 — Final Experimental Framework

## Objective

Run the controlled experiment that addresses the research questions.

Use a held-out test set.

For each test circuit, run:

1.  Random optimizer.
2.  Greedy optimizer.
3.  GNN-guided optimizer.

Optional:

4.  Aggregate-feature MLP optimizer.

## Fair comparison

Use comparable:

- circuit sets;
- transformation rules;
- search budgets;
- equivalence checker;
- stopping conditions.

## Record

``` text
circuit_id
method
initial_gate_count
final_gate_count
initial_cnot_count
final_cnot_count
initial_depth
final_depth
gate_reduction
cnot_reduction
depth_reduction
candidate_evaluations
equivalence_checks
iterations
runtime
```

------------------------------------------------------------------------

# PHASE 19 — Ablation Studies

## Objective

Determine which components contribute to performance.

Minimum comparison:

### A — Random candidate selection

### B — Greedy candidate selection

### C — Aggregate-feature ML

### D — Graph-based GNN

This tests whether structural graph information provides useful
information beyond aggregate statistics.

Optional later studies:

- with vs. without feedback;
- transformation subsets;
- different graph edge definitions.

------------------------------------------------------------------------

# PHASE 20 — Final Validation and Reproducibility

## Objective

Perform a clean end-to-end run from the repository.

The run must not depend on hidden notebook state.

Document:

- setup;
- dataset generation;
- training;
- evaluation;
- configuration.

## Record

- random seed;
- software versions;
- configuration;
- dataset version;
- model configuration;
- transformation library version.

## Final correctness checks

For every optimized circuit:

1.  Reconstruct original.
2.  Reconstruct optimized circuit.
3.  Verify equivalence.
4.  Recalculate metrics.
5.  Compare with logged metrics.
6.  Save final result.

------------------------------------------------------------------------

# PHASE 21 — Final Demonstration

The final demonstration should show an unseen circuit and the complete
decision pipeline.

``` text
Original Circuit
      |
      v
Candidate Generation
      |
      v
GNN Ranking
      |
      v
Exact Verification
      |
      v
Cost Evaluation
      |
      v
Next Iteration
      |
      v
Optimized Circuit
```

Final report:

``` text
Metric          Original    Optimized
--------------------------------------
Gate count          X           Y
CNOT count          X           Y
Depth               X           Y
Equivalent          --          YES
```

The actual values must come from experiments.

------------------------------------------------------------------------

# 22. Research Questions

The final system should investigate:

### RQ1

Can a GNN learn a meaningful relationship between quantum-circuit
structure and the benefit of valid transformations?

### RQ2

Does graph-based representation provide useful information beyond
aggregate circuit statistics?

### RQ3

Can GNN-guided candidate ranking improve search compared with random
selection?

### RQ4

How does GNN-guided optimization compare with greedy optimization under
a comparable evaluation budget?

### RQ5

Can verified optimization outcomes be used as feedback to improve
subsequent candidate ranking?

These are hypotheses to test, not predetermined conclusions.

------------------------------------------------------------------------

# 23. Definition of Project Success

The project is successful if it demonstrates:

- reliable small-circuit generation;
- exact circuit reconstruction;
- deterministic cost measurement;
- valid transformation generation;
- independent equivalence verification;
- clean ground-truth dataset construction;
- meaningful graph representation;
- GNN training on verified observations;
- candidate ranking;
- end-to-end GNN-guided optimization;
- safe acceptance of only verified candidates;
- random and greedy baselines;
- reproducible experiments.

The project does not require:

- global optimality;
- superiority on every circuit;
- large-scale circuits;
- physical quantum hardware;
- perfect prediction accuracy.

------------------------------------------------------------------------

# 24. Debugging Order

If something fails, debug in this order:

1.  Quantum representation.
2.  Equivalence verification.
3.  Transformation rules.
4.  Cost evaluator.
5.  Candidate generator.
6.  Dataset.
7.  Graph representation.
8.  GNN.
9.  Optimizer.

Do not immediately make the neural network larger when an experiment
fails.

First identify whether the deterministic pipeline is correct.

------------------------------------------------------------------------

# 25. Critical Conceptual Distinctions

## Circuit generation vs. candidate generation

Initial generation creates the starting problem.

Candidate generation modifies an existing circuit using known
transformations.

## Prediction vs. verification

The GNN predicts.

Qiskit verifies.

They must remain separate.

## Optimization vs. synthesis

Synthesis:

``` text
Desired operation -> circuit
```

This project:

``` text
Existing circuit -> cheaper equivalent circuit
```

## Structure vs. aggregate statistics

Aggregate features describe:

``` text
gate count
depth
CNOT count
```

Graph representation describes:

``` text
gate relationships
qubit interactions
operation dependencies
ordering
```

The project investigates whether this structural information helps
optimization.

------------------------------------------------------------------------

# 26. Minimum Viable Project

If time is limited, prioritize:

``` text
Phase 1  Environment
Phase 2  Circuit Representation
Phase 3  Circuit Generator
Phase 4  Cost Metrics
Phase 5  Equivalence Checker
Phase 6  Transformation Library
Phase 7  Candidate Generator
Phase 8  Ground-Truth Dataset
Phase 10 Graph Representation
Phase 11 Random + Greedy Baselines
Phase 13 GNN
Phase 14 GNN Training
Phase 15 GNN Ranking
Phase 16 Iterative Optimizer
```

Feedback learning and extensive ablation studies should follow only
after the core system is stable.

------------------------------------------------------------------------

# 27. Non-Negotiable Final Architecture

``` text
                 EXISTING CIRCUIT
                         |
                         v
                 CIRCUIT ANALYSIS
                         |
                         v
                 GRAPH REPRESENTATION
                         |
                         v
                 PROCEDURAL SEARCH
                 Candidate Generator
                         |
                         v
                       GNN
                 Candidate Ranking
                         |
                         v
                 EXACT VERIFICATION
                       Qiskit
                         |
                         v
                   COST EVALUATION
                         |
                    beneficial?
                     /                          NO         YES
                   |           |
                 reject        v
                         New Current Circuit
                                |
                                v
                         Repeat Optimization

                                |
                                v
                       Verified Observations
                                |
                                v
                         Training Dataset
                                |
                                v
                            GNN Update
```

## Final design principle

> **The procedural system explores the search space, the GNN prioritizes
> the search, and quantum verification determines the truth.**
