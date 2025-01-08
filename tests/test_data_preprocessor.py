import pytest
import pandas as pd
import numpy as np
from src.data_preprocessor import preprocessor_data

@pytest.fixture
def sample_data():
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

def test_preprocessor_data_success(sample_data):
    columns_to_impute = ['trestbps', 'chol', 'thalach', 'oldpeak']
    result = preprocessor_data(sample_data, columns_to_impute)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert all(col in result.columns for col in columns_to_impute)

def test_preprocessor_data_with_zeros(sample_data):
    # Modify sample data to include zeros
    sample_data['trestbps'] = 0
    sample_data['chol'] = 0
    
    columns_to_impute = ['trestbps', 'chol', 'thalach', 'oldpeak']
    result = preprocessor_data(sample_data, columns_to_impute)
    
    assert pd.isna(result['trestbps'].iloc[0])
    assert pd.isna(result['chol'].iloc[0])

def test_preprocessor_data_invalid_input():
    invalid_data = "not a dictionary"
    columns_to_impute = ['trestbps', 'chol']
    
    with pytest.raises(Exception):
        preprocessor_data(invalid_data, columns_to_impute)