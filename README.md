# Road Accident Severity Prediction Using Machine Learning

## Project Overview
This project predicts the severity class of a road accident using machine learning. It uses road and environmental conditions, vehicle count, number of casualties, and other accident-related features to make a prediction through a Streamlit web application.

## Objectives
- Predict road accident severity from input features.
- Apply data preprocessing and categorical feature encoding.
- Address class imbalance using SMOTENC on the training data.
- Provide an interactive interface for users to enter accident details and view predictions.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- SMOTENC (imbalanced-learn)
- Random Forest Classifier
- Streamlit
- Joblib

## Model
A Random Forest classifier was trained using accident data. The current model achieved approximately **61.61% accuracy** on the held-out test set. Performance varies across severity classes, particularly because the dataset contains an imbalanced distribution of accident severity labels.

## Application
The Streamlit application accepts accident-related inputs and displays a predicted severity class and model probabilities.

## Project Files
- `app (1).py` — Streamlit application source code.
- `accident_severity_model_compressed.pkl` — Compressed trained model.
- `accident_preprocessor (1).pkl` — Saved preprocessing pipeline.
- `requirements.txt` — Required Python packages.

## How to Run Locally
1. Install Python.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run "app (1).py"
   ```

## Limitations
This is an educational machine learning project, not a validated road-safety or emergency decision-making system. Predictions depend on the training data and may be less reliable for underrepresented severity classes. The predicted class codes should be interpreted using the dataset's official documentation.

## Future Improvements
- Improve performance on underrepresented severity classes.
- Evaluate additional machine learning algorithms.
- Validate predictions on independent datasets.
- Improve the user interface and model evaluation.
