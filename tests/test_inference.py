import pytest
import pandas as pd
import joblib
from src.inference import main
from src.model_loader import load_model
from src.data_preprocessor import preprocessor_data

@pytest.fixture
def sample_input_data():
    return [
        {
            "age": 67, "sex": 1, "cp": 4, "trestbps": 120, "chol": 237,
            "fbs": 0, "restecg": 0, "thalach": 71, "exang": 0, "oldpeak": 1.0,
            "slope": 2, "ca": 0.0, "thal": 3.0
        }
    ]

@pytest.fixture
def mock_model(tmp_path):
    # Create a mock model that always predicts 1
    class MockModel:
        def predict(self, X):
            return [1]
    
    model = MockModel()
    model_path = tmp_path / "trained_model_2025-01-08.joblib"
    joblib.dump(model, model_path)
    return str(model_path)

def test_inference_prediction_format(sample_input_data, mock_model, monkeypatch):
    # Mock the model path
    def mock_load_model(path):
        return joblib.load(mock_model)
    
    monkeypatch.setattr("src.inference.load_model", mock_load_model)
    
    # Process single input data
    columns_to_use = ['trestbps', 'chol', 'thalach', 'oldpeak']
    preprocessed_data = preprocessor_data(sample_input_data[0], columns_to_use)
    
    # Load model and make prediction
    model = load_model(mock_model)
    prediction = model.predict(preprocessed_data)
    
    # Verify prediction format
    assert isinstance(prediction, list)
    assert all(isinstance(pred, (int, np.int64)) for pred in prediction)
    assert all(pred in [0, 1] for pred in prediction)

def test_inference_expected_prediction(sample_input_data, mock_model, monkeypatch):
    # Mock the model path
    def mock_load_model(path):
        return joblib.load(mock_model)
    
    monkeypatch.setattr("src.inference.load_model", mock_load_model)
    
    # Process single input data
    columns_to_use = ['trestbps', 'chol', 'thalach', 'oldpeak']
    preprocessed_data = preprocessor_data(sample_input_data[0], columns_to_use)
    
    # Load model and make prediction
    model = load_model(mock_model)
    prediction = model.predict(preprocessed_data)
    
    # Verify prediction is 1 (as per our mock model)
    assert prediction == [1]