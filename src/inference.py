import sys
import os
# Agregar la raíz del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_loader import load_model
from src.data_preprocessor import preprocessor_data

def main():
    # Load the model
    model_path = "models/trained_model_2025-01-08.joblib"
    model = load_model(model_path)
    print("paso la carga del modelo")

    # Sample predictions
    input_data = [
    {"age": 67, "sex": 1, "cp": 4, "trestbps": 120, "chol": 237, "fbs": 0, "restecg": 0, "thalach": 71, "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 0.0, "thal": 3.0},
    {"age": 58, "sex": 1, "cp": 4, "trestbps": 100, "chol": 234, "fbs": 0, "restecg": 0, "thalach": 156, "exang": 0, "oldpeak": 0.1, "slope": 1, "ca": 1.0, "thal": 7.0},
    {"age": 47, "sex": 1, "cp": 4, "trestbps": 110, "chol": 275, "fbs": 0, "restecg": 2, "thalach": 118, "exang": 1, "oldpeak": 1.0, "slope": 2, "ca": 1.0, "thal": 3.0},
    {"age": 52, "sex": 1, "cp": 4, "trestbps": 125, "chol": 212, "fbs": 0, "restecg": 0, "thalach": 168, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 2.0, "thal": 7.0},
    {"age": 58, "sex": 1, "cp": 4, "trestbps": 146, "chol": 218, "fbs": 0, "restecg": 0, "thalach": 105, "exang": 0, "oldpeak": 2.0, "slope": 2, "ca": 1.0, "thal": 7.0},
    {"age": 61, "sex": 1, "cp": 4, "trestbps": 138, "chol": 166, "fbs": 0, "restecg": 2, "thalach": 125, "exang": 1, "oldpeak": 3.6, "slope": 2, "ca": 1.0, "thal": 3.0},
    {"age": 42, "sex": 1, "cp": 4, "trestbps": 136, "chol": 315, "fbs": 0, "restecg": 0, "thalach": 125, "exang": 1, "oldpeak": 1.8, "slope": 2, "ca": 0.0, "thal": 6.0},
    {"age": 52, "sex": 1, "cp": 4, "trestbps": 128, "chol": 204, "fbs": 1, "restecg": 0, "thalach": 156, "exang": 1, "oldpeak": 1.0, "slope": 2, "ca": 0.0, "thal": 3.0},
    {"age": 59, "sex": 1, "cp": 3, "trestbps": 126, "chol": 218, "fbs": 1, "restecg": 0, "thalach": 134, "exang": 0, "oldpeak": 2.2, "slope": 2, "ca": 1.0, "thal": 6.0},
    {"age": 40, "sex": 1, "cp": 4, "trestbps": 152, "chol": 223, "fbs": 0, "restecg": 0, "thalach": 181, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0.0, "thal": 7.0}
]


    # Columns to use
    columns_to_use = ['trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Preprocess the data
    preprocessed_data = [preprocessor_data(data=data, columns_to_impute=columns_to_use) for data in input_data]
    print("hizo el preprocesamiento con éxito")

    try: 
        predictions = [model.predict(data) for data in preprocessed_data]
        for i, prediction in enumerate(predictions):
            print(f"Prediction for input data {i+1}: {prediction}")
    except Exception as e:
        print(f"Error running inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
