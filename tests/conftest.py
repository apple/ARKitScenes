import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest
import numpy as np
import torch


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that is cleaned up after the test."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> dict:
    """Provide a mock configuration dictionary for testing."""
    return {
        "model": {
            "name": "test_model",
            "num_classes": 10,
            "input_channels": 3,
            "learning_rate": 0.001,
        },
        "data": {
            "batch_size": 32,
            "num_workers": 4,
            "train_path": "/path/to/train",
            "val_path": "/path/to/val",
        },
        "training": {
            "epochs": 100,
            "checkpoint_dir": "/path/to/checkpoints",
            "log_interval": 10,
        },
    }


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample image array for testing."""
    return np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_depth_map() -> np.ndarray:
    """Create a sample depth map for testing."""
    return np.random.rand(480, 640).astype(np.float32) * 10.0


@pytest.fixture
def sample_point_cloud() -> np.ndarray:
    """Create a sample 3D point cloud for testing."""
    num_points = 1000
    points = np.random.randn(num_points, 3).astype(np.float32)
    return points


@pytest.fixture
def sample_torch_tensor() -> torch.Tensor:
    """Create a sample PyTorch tensor for testing."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock PyTorch model for testing."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.train = MagicMock(return_value=model)
    model.forward = MagicMock(return_value=torch.randn(1, 10))
    model.parameters = MagicMock(return_value=[torch.randn(10, 10)])
    return model


@pytest.fixture
def mock_dataset() -> MagicMock:
    """Create a mock dataset for testing."""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=100)
    dataset.__getitem__ = MagicMock(
        return_value=(torch.randn(3, 224, 224), torch.tensor(1))
    )
    return dataset


@pytest.fixture
def mock_dataloader(mock_dataset) -> MagicMock:
    """Create a mock dataloader for testing."""
    dataloader = MagicMock()
    dataloader.__iter__ = MagicMock(
        return_value=iter([(torch.randn(32, 3, 224, 224), torch.randint(0, 10, (32,)))])
    )
    dataloader.dataset = mock_dataset
    return dataloader


@pytest.fixture
def sample_bounding_boxes() -> np.ndarray:
    """Create sample bounding boxes for 3D object detection testing."""
    # Format: [x_center, y_center, z_center, width, height, depth, rotation]
    boxes = np.array([
        [0.0, 0.0, 5.0, 2.0, 1.5, 4.0, 0.0],
        [3.0, 0.0, 10.0, 2.5, 1.8, 4.5, np.pi/4],
        [-2.0, 0.0, 8.0, 1.8, 1.6, 3.8, -np.pi/6],
    ], dtype=np.float32)
    return boxes


@pytest.fixture
def sample_lidar_data() -> np.ndarray:
    """Create sample LiDAR data for testing."""
    num_points = 5000
    # Generate points in a cone pattern to simulate LiDAR
    angles = np.random.uniform(0, 2*np.pi, num_points)
    distances = np.random.uniform(0.5, 50.0, num_points)
    heights = np.random.uniform(-2.0, 2.0, num_points)
    
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    z = heights
    intensity = np.random.uniform(0, 1, num_points)
    
    return np.column_stack([x, y, z, intensity]).astype(np.float32)


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture log messages during tests."""
    with caplog.at_level("DEBUG"):
        yield caplog


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Disable GPU for tests by default
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Add integration marker to tests in integration directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests with "slow" in their name
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)