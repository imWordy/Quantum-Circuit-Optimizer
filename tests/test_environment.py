"""Test environment, dependencies, Qiskit simulator, PyTorch, PyG, and seed reproducibility."""

import pytest
import numpy as np
import torch
from torch_geometric.data import Data

import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qco.utils.reproducibility import set_seed
from qco.utils.config import load_config
from qco.utils.logging import setup_logger


def test_dependency_imports():
    """Verify all required Phase 1 libraries can be imported successfully."""
    import pandas
    import sklearn
    import scipy
    import matplotlib

    assert qiskit.__version__ is not None
    assert torch.__version__ is not None
    assert pandas.__version__ is not None
    assert sklearn.__version__ is not None
    assert scipy.__version__ is not None
    assert matplotlib.__version__ is not None


def test_config_loading():
    """Verify central configuration loads properly with expected keys."""
    cfg = load_config()
    assert "seed" in cfg
    assert "circuit" in cfg
    assert "verification" in cfg
    assert "optimization" in cfg
    assert "data" in cfg
    assert cfg["circuit"]["min_qubits"] <= cfg["circuit"]["max_qubits"]


def test_qiskit_circuit_and_simulation():
    """Verify Qiskit circuit creation and AerSimulator execution."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    backend = AerSimulator()
    job = backend.run(qc, shots=100)
    result = job.result()
    counts = result.get_counts()

    # Bell state produces mainly '00' and '11'
    assert "00" in counts or "11" in counts
    assert sum(counts.values()) == 100


def test_pytorch_tensor_creation():
    """Verify PyTorch tensor creation and basic operations."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = x @ x.T
    assert y.shape == (2, 2)
    assert not torch.isnan(y).any()


def test_pytorch_geometric_graph_creation():
    """Verify PyTorch Geometric graph data object creation."""
    # Graph with 3 nodes and 2 directed edges
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    assert data.num_nodes == 3
    assert data.num_edges == 2
    assert data.validate(raise_on_error=True)


def test_seed_reproducibility():
    """Verify random seed reproducibility across random, numpy, and torch."""
    set_seed(1234)
    np_rand1 = np.random.rand(5)
    torch_rand1 = torch.rand(5)

    set_seed(1234)
    np_rand2 = np.random.rand(5)
    torch_rand2 = torch.rand(5)

    np.testing.assert_allclose(np_rand1, np_rand2)
    assert torch.allclose(torch_rand1, torch_rand2)


def test_logging():
    """Verify logger setup runs without errors."""
    logger = setup_logger("test_logger")
    logger.info("Test log message.")
    assert logger.name == "test_logger"
