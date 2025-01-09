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
    """Test successful data preprocessing with valid input."""
    columns_to_impute = ['trestbps', 'chol', 'thalach', 'oldpeak']
    result = preprocessor_data(sample_data, columns_to_impute)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert all(col in result.columns for col in columns_to_impute)
    assert result['trestbps'].iloc[0] == 120
    assert result['chol'].iloc[0] == 237

def test_preprocessor_data_with_zeros(sample_data):
    """Test zero value imputation."""
    sample_data['trestbps'] = 0
    sample_data['chol'] = 0
    
    columns_to_impute = ['trestbps', 'chol', 'thalach', 'oldpeak']
    result = preprocessor_data(sample_data, columns_to_impute)
    
    assert pd.isna(result['trestbps'].iloc[0])
    assert pd.isna(result['chol'].iloc[0])
    assert not pd.isna(result['thalach'].iloc[0])

def test_preprocessor_data_column_presence(sample_data):
    """Test all expected columns are present in output."""
    columns_to_impute = ['trestbps', 'chol', 'thalach', 'oldpeak']
    result = preprocessor_data(sample_data, columns_to_impute)
    
    expected_columns = list(sample_data.keys())
    assert all(col in result.columns for col in expected_columns)

def test_preprocessor_data_invalid_input():
    """Test handling of invalid input data."""
    invalid_inputs = [
        "not a dictionary",
        None,
        42,
        [],
        {'invalid_key': 'value'}
    ]
    
    columns_to_impute = ['trestbps', 'chol']
    for invalid_input in invalid_inputs:
        with pytest.raises(Exception):
            preprocessor_data(invalid_input, columns_to_impute)