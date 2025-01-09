# XGBoost Inference Pipeline for Heart Disease UCI

## Project Overview
This project implements an inference pipeline for heart disease prediction using a pre-trained XGBoost model. It provides functionality for loading the model, preprocessing new data, and making predictions.

## Project Structure
```
gillopy-Deployment_XGBoost_Inference_Heart_Disease_UCI/
├── README.md                    # Project documentation
├── Dockerfile                   # Docker container configuration
├── LICENSE                      # Project license
├── pyproject.toml              # Poetry dependency management
├── .dockerignore               # Docker build exclusions
├── models/                     # Pre-trained model files
│   ├── trained_model_2025-01-06.joblib
│   └── trained_model_2025-01-08.joblib
├── src/                        # Source code
│   ├── data_preprocessor.py    # Data preprocessing functionality
│   ├── inference.py           # Main inference pipeline
│   └── model_loader.py        # Model loading utilities
└── tests/                      # Test suite
    ├── __init__.py
    ├── test_data_preprocessor.py
    ├── test_inference.py
    └── test_model_loader.py
```

## Requirements

### Python Version
- Python 3.10 (specific version requirement)

### Dependencies
All dependencies are managed through Poetry and specified in pyproject.toml:
- pandas (^2.2.3)
- scikit-learn (^1.6.0)
- xgboost (^2.1.3)
- joblib (^1.4.2)

## Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/gillopy/Deployment_XGBoost_Inference_Heart_Disease_UCI
   cd Deployment_XGBoost_Inference_Heart_Disease_UCI
   ```

2. **Install Dependencies**:
   ```bash
   poetry install
   ```

3. **Docker Setup** (optional):
   ```bash
   docker build -t heart-disease-inference .
   ```

## Usage

### Running Inference
Execute the main inference script:
```bash
poetry run python src/inference.py
```

### Input Data Format
The model expects input data in the following format:
```python
{
    "age": int,
    "sex": int,
    "cp": int,
    "trestbps": int,
    "chol": int,
    "fbs": int,
    "restecg": int,
    "thalach": int,
    "exang": int,
    "oldpeak": float,
    "slope": int,
    "ca": float,
    "thal": float
}
```

## Testing
Run the test suite using pytest:
```bash
poetry run pytest
```

The test suite includes:
- Data preprocessing validation
- Inference pipeline testing
- Model loading verification

## Project Components

### Data Preprocessor
- Handles missing value imputation
- Converts input dictionary to DataFrame format
- Implements data validation checks

### Model Loader
- Loads the pre-trained XGBoost model
- Includes error handling for missing model files
- Validates model compatibility

### Inference Pipeline
- Orchestrates the complete inference process
- Supports batch predictions
- Provides formatted output

## Docker Support
The project includes Docker support for containerized deployment:
- Base Python 3.10 image
- Automatic dependency installation
- Environment isolation

## License
Apache License 2.0

## Author
Guillermo (guillermocabrera9710@gmail.com)
