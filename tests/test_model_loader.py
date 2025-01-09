import pytest
import os
import joblib
import numpy as np
from src.model_loader import load_model

@pytest.fixture
def mock_model_path(tmp_path):
    """Create a mock model file for testing."""
    class MockModel:
        def predict(self, X):
            return np.array([1])
    
    model = MockModel()
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)
    return str(model_path)

def test_load_model_success(mock_model_path):
    """Test successful model loading."""
    model = load_model(mock_model_path)
    assert hasattr(model, 'predict')
    
    # Test prediction functionality
    test_input = pd.DataFrame({'feature': [1]})
    prediction = model.predict(test_input)
    assert isinstance(prediction, np.ndarray)
    assert prediction[0] in [0, 1]

def test_load_model_file_not_found():
    """Test handling of non-existent model file."""
    with pytest.raises(Exception):
        load_model("nonexistent_model.joblib")

def test_load_model_corrupted_file(tmp_path):
    """Test handling of corrupted model file."""
    corrupt_path = tmp_path / "corrupt_model.joblib"
    with open(corrupt_path, 'w') as f:
        f.write("Not a valid joblib file")
    
    with pytest.raises(Exception):
        load_model(str(corrupt_path))