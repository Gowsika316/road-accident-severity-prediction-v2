🚗 AI-Based Road Accident Severity Prediction

An interactive machine learning application that predicts road accident severity from road, vehicle, junction, and environmental information.

🌐 Live Demo

Try the application: Road Accident Severity Prediction

📌 Project Overview

Road accidents can result in injuries and loss of life. Analysing accident-related factors can help identify patterns associated with different levels of accident severity.

This project uses machine learning to predict an accident severity class based on user-provided accident details. A Streamlit web application provides an interactive interface for entering input values and viewing the model's prediction and estimated class probabilities.

🎯 Objectives

Analyse road accident data using machine learning.

Preprocess accident data for model training.

Address class imbalance in the training dataset.

Predict accident severity from selected input features.

Deploy the trained model as an interactive web application.

🛠️ Technologies Used

Programming Language: Python

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn, Random Forest

Class Imbalance Handling: SMOTENC

Model Serialization: Joblib

Web Application: Streamlit

Version Control: Git and GitHub

📊 Dataset

The project uses the India Road Accident Dataset, including the AccidentsBig.csv dataset file.

The model uses accident-related features such as:

Number of vehicles and casualties

Day of the week

Road class and road type

Speed limit

Junction details and control

Light and weather conditions

Road surface conditions

Carriageway hazards

Urban or rural area

Police attendance at the accident scene

The target variable is Accident_Severity.

⚙️ Project Workflow

Data collection: Load the road accident dataset.

Data preprocessing: Handle missing values, remove duplicates, and prepare the input features.

Feature preparation: Encode categorical features and scale numerical features where appropriate.

Class imbalance handling: Apply SMOTENC to the training data to address the imbalance among severity classes.

Model training: Train a Random Forest classifier.

Model evaluation: Evaluate predictions on held-out test data using appropriate classification metrics.

Model serialization: Save the trained model and preprocessing pipeline using Joblib.

Deployment: Load the saved artifacts in a Streamlit application and display predictions.

🖥️ Application Features

Interactive form for entering accident details

Prediction of the accident severity class

Estimated probability for each severity class

Web-based interface accessible through a browser

🧰 How to Run Locally

1. Clone the repository

git clone https://github.com/Gowsika316/road-accident-severity-prediction-v2.git
cd road-accident-severity-prediction-v2

2. Install dependencies

pip install -r requirements.txt

3. Run the Streamlit application

streamlit run "app (1).py"

Ensure the saved model and preprocessing files are present in the repository's root directory.

📈 Model Evaluation

The model should be evaluated using metrics such as:

Accuracy

Precision

Recall

F1-score

Confusion matrix

The final evaluation results should be added after confirming them against the held-out test dataset. Performance can vary across severity classes, particularly when the original dataset is imbalanced.

⚠️ Limitations

Predictions depend on the quality and coverage of the training dataset.

Class imbalance and differences in performance across severity classes can affect reliability.

Model probability estimates are not guarantees of real-world outcomes.

This application is an academic prototype and should not be used as the sole basis for real-world emergency or road-safety decisions.

🔮 Future Enhancements

Improve minority-class prediction performance.

Compare multiple machine learning algorithms.

Add more comprehensive model evaluation and explainability.

Integrate reliable real-time traffic and road-condition data, where available.

Explore deployment and monitoring improvements.
