import pytest
import pandas as pd
import numpy as np
import joblib
from src.inference import main
from src.model_loader import load_model
from src.data_preprocessor import preprocessor_data

@pytest.fixture
def sample_input_data():
    """Provide sample input data for testing."""
    return {
        "age": 67,
        "sex": 1,
        "cp": 4,
        "trestbps": 120,
        "chol": 237,
        "fbs": 0,
        "restecg": 0,
        "thalach": 71,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0.0,
        "thal": 3.0
    }

@pytest.fixture
def mock_model(tmp_path):
    """Create a mock model that returns predetermined predictions."""
    class MockModel:
        def predict(self, X):
            return np.array([1])
    
    model = MockModel()
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)
    return str(model_path)

def test_end_to_end_inference(sample_input_data, mock_model, monkeypatch):
    """Test the complete inference pipeline."""
    def mock_load_model(path):
        return joblib.load(mock_model)
    
    monkeypatch.setattr("src.inference.load_model", mock_load_model)
    
    # Process data
    columns_to_use = ['trestbps', 'chol', 'thalach', 'oldpeak']
    preprocessed_data = preprocessor_data(sample_input_data, columns_to_use)
    
    # Load model and predict
    model = load_model(mock_model)
    prediction = model.predict(preprocessed_data)
    
    # Verify prediction
    assert isinstance(prediction, np.ndarray)
    assert prediction[0] in [0, 1]
    assert prediction[0] == 1  # Expected output for mock model

def test_inference_with_edge_cases(mock_model, monkeypatch):
    """Test inference with edge case inputs."""
    edge_cases = [
        {
            "age": 0,  # Minimum age
            "sex": 1,
            "cp": 4,
            "trestbps": 0,  # Will be imputed
            "chol": 0,      # Will be imputed
            "fbs": 0,
            "restecg": 0,
            "thalach": 0,   # Will be imputed
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 1,
            "ca": 0.0,
            "thal": 3.0
        },
        {
            "age": 120,  # Maximum age
            "sex": 0,
            "cp": 1,
            "trestbps": 200,
            "chol": 500,
            "fbs": 1,
            "restecg": 2,
            "thalach": 200,
            "exang": 1,
            "oldpeak": 6.0,
            "slope": 3,
            "ca": 3.0,
            "thal": 7.0
        }
    ]
    
    def mock_load_model(path):
        return joblib.load(mock_model)
    
    monkeypatch.setattr("src.inference.load_model", mock_load_model)
    
    for case in edge_cases:
        columns_to_use = ['trestbps', 'chol', 'thalach', 'oldpeak']
        preprocessed_data = preprocessor_data(case, columns_to_use)
        model = load_model(mock_model)
        prediction = model.predict(preprocessed_data)
        
        assert isinstance(prediction, np.ndarray)
        assert prediction[0] in [0, 1]