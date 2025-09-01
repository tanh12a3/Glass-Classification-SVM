so you have got the idea, point out what is wrong with the code
import tkinter as tk
from tkinter import messagebox

import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Load dataset and preprocess
def load_data(file_path):
    """Load dataset and split into features and labels."""
    data = pd.read_csv(file_path)
    features = data.iloc[:, :-1]  # Extract feature columns
    labels = data.iloc[:, -1]  # Extract label column
    return features, labels


# Train multiple SVM models with random splits
def train_bagging_models(X, y, n_splits=10):
    """Train multiple SVM models with random 80-20 splits."""
    models = []  # List to store trained models
    scalers = []  # List to store corresponding scalers

    for i in range(n_splits):
        # Create random split for training
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y
        )

        # Convert to NumPy array to avoid feature name warnings
        X_train_np = X_train.values

        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_np)

        # Train SVM
        model = SVC(
            kernel="rbf", C=100, gamma=0.1, probability=True, class_weight="balanced"
        )
        model.fit(X_train_scaled, y_train)

        # Save the model and scaler
        models.append(model)
        scalers.append(scaler)

        print(f"Model {i + 1} trained.")

    return models, scalers


# Predict using majority voting
def majority_vote(models, scalers, X_test):
    """Make predictions using all models and perform majority voting."""
    # Convert test data to NumPy array
    X_test_np = X_test.values

    # Collect predictions from each model
    all_predictions = [
        model.predict(scaler.transform(X_test_np))
        for model, scaler in zip(models, scalers)
    ]

    # Convert to NumPy array for majority voting
    predictions_array = np.array(all_predictions)

    # Perform majority voting
    voted_predictions = mode(predictions_array, axis=0).mode.flatten()

    return voted_predictions


# Evaluate models
def evaluate(models, scalers, X_test, y_test):
    """Evaluate the ensemble of models."""
    predictions = majority_vote(models, scalers, X_test)
    accuracy = accuracy_score(y_test, predictions) * 100

    print(f"Accuracy: {accuracy:.2f}%")
    print("Classification Report:\n", classification_report(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))




# Main function
def main():
    file_path = "glass.csv"

    # Load dataset
    X, y = load_data(file_path)

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train models
    models, scalers = train_bagging_models(X, y)

    # Evaluate the ensemble
    evaluate(models, scalers, X_test, y_test)

    # Launch GUI
    create_gui(models, scalers)


if __name__ == "__main__":
    main() 