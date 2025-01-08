import sys
import os
# Agregar la raíz del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_loader import load_model
from src.data_preprocessor import preprocessor_data

def main():
    # Load the model
    model_path = "models/trained_model_2025-01-06.joblib"
    model = load_model(model_path)
    print("paso la carga del modelo")

    # Sample predictions
    input_data = [
        {
            "age": 48,
            "sex": 1,
            "cp": 2,
            "trestbps": 110,
            "chol": 229,
            "fbs": 0,
            "restecg": 0,
            "thalach": 168,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 3,
            "ca": 0.0,
            "thal": 7.0
        }
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
