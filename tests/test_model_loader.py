import pytest
import os
import joblib
from src.model_loader import load_model

def test_load_model_success(tmp_path):
    # Create a mock model file
    mock_model = {"mock": "model"}
    model_path = os.path.join(tmp_path, "test_model.joblib")
    joblib.dump(mock_model, model_path)
    
    # Test loading the model
    loaded_model = load_model(model_path)
    assert loaded_model == mock_model

def test_load_model_file_not_found():
    with pytest.raises(Exception):
        load_model("nonexistent_model.joblib")