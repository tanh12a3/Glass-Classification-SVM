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
    print("Class distribution:")
    print(labels.value_counts())
    return features, labels


# Train multiple SVM models with matched train-test splits
def train_bagging_models(X, y, n_splits=10):
    """Train multiple SVM models with matched train-test splits."""
    model_data = []  # List to store (model, scaler, X_test, y_test)

    for i in range(n_splits):
        # Create random split for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y
        )

        # Convert to NumPy array to avoid feature name warnings
        X_train_np = X_train.values
        X_test_np = X_test.values

        # Scale training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_np)

        # Train SVM rbf~Gaussian dist, g=0.1 -> small region of similar
        model = SVC(
            kernel="rbf", C=100, gamma=0.1, probability=True, class_weight="balanced"
        )
        model.fit(X_train_scaled, y_train)

        # Save model, scaler, and corresponding test set
        model_data.append((model, scaler, X_test_np, y_test))
        print(f"Model {i + 1} trained.")

    return model_data


# Evaluate each model with its corresponding test data
def evaluate_individual_models(models_data):
    """Evaluate each model independently with its corresponding test data."""
    for i, (model, scaler, X_test, y_test) in enumerate(models_data):
        # Scale test data
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)  # test the model

        # Calculate accuracy for this model
        accuracy = accuracy_score(y_test, predictions) * 100
        print(f"Model {i + 1} Accuracy: {accuracy:.2f}%")


def evaluate_ensemble(models_data):
    """
    Evaluate the ensemble by passing each test set through all models,
    performing majority voting, and comparing predictions with true labels.
    """
    all_y_test = []  # Store all true labels
    final_predictions = []  # Store final majority-voted predictions

    for model_index, (_, _, X_test, y_test) in enumerate(models_data):
        print(f"\nEvaluating Test Data from Model {model_index + 1}")
        # Collect predictions for the current test data across all models
        test_predictions = []

        for i, (model, scaler, _, _) in enumerate(models_data):
            # Scale the current test data using each model's scaler
            X_test_scaled = scaler.transform(X_test)
            predictions = model.predict(X_test_scaled)
            test_predictions.append(predictions)

            # Debugging output for each prediction
            print(f"Model {i + 1}: Predicted {predictions}")

        # Perform majority voting on the predictions
        test_predictions = np.array(test_predictions)
        voted_predictions = mode(
            test_predictions, axis=0
        ).mode.flatten()  # find majority

        # Store results
        final_predictions.extend(voted_predictions)  # store the final vote to array
        all_y_test.extend(y_test)  # corresponding test data

        print(f"True Labels for Model {model_index + 1}: {y_test.tolist()}")
        print(f"Voted Prediction for Model {model_index + 1}: {voted_predictions}")

        # print(f"True Label for Model {model_index + 1}: {y_test}")

        print(
            f"y_test Shape: {len(y_test)}, Voted Predictions Shape: {len(voted_predictions)}"
        )

    # Convert lists to numpy arrays
    all_y_test = np.array(all_y_test)
    final_predictions = np.array(final_predictions)

    # Evaluate the majority-voted predictions
    accuracy = accuracy_score(all_y_test, final_predictions) * 100
    print(f"\nFinal Ensemble Accuracy: {accuracy:.2f}%")
    print(
        "Classification Report:\n", classification_report(all_y_test, final_predictions)
    )
    print("Confusion Matrix:\n", confusion_matrix(all_y_test, final_predictions))


import tkinter as tk


def create_gui(models_data):
    """
    Create a simple GUI for entering random inputs and predicting glass type using the ensemble.
    """

    def predict():
        try:
            # Get user inputs from GUI
            inputs = [[float(entry.get()) for entry in entries]]

            print(f"\nInputs: {inputs}")

            # Convert input to NumPy array
            inputs_np = np.array(inputs)

            # Collect predictions from all models
            all_predictions = []
            for i, (model, scaler, _, _) in enumerate(models_data):
                # Scale input using the scaler of each model
                scaled_input = scaler.transform(inputs_np)
                prediction = model.predict(scaled_input)
                all_predictions.append(prediction)

                # Debugging output
                print(f"Model {i + 1}: Prediction {prediction[0]}")

            # Perform majority voting
            all_predictions = np.array(all_predictions)
            voted_prediction = mode(all_predictions, axis=0).mode.flatten()[0]

            # Display result
            result_label.config(text=f"Predicted Glass Type: {int(voted_prediction)}")

        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    # Create GUI window
    window = tk.Tk()
    window.title("Glass Type Predictor")

    # Define input field labels
    feature_names = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
    entries = []

    # Create input fields dynamically
    for i, feature in enumerate(feature_names):
        tk.Label(window, text=feature).grid(row=i, column=0, padx=5, pady=5)
        entry = tk.Entry(window)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entries.append(entry)

    # Create predict button
    predict_button = tk.Button(window, text="Predict", command=predict)
    predict_button.grid(row=len(feature_names), column=0, columnspan=2, pady=10)

    # Create result label
    result_label = tk.Label(window, text="")
    result_label.grid(row=len(feature_names) + 1, column=0, columnspan=2, pady=10)

    # Start the GUI event loop
    window.mainloop()


# Main function update to include GUI
def main():
    file_path = "glass.csv"

    # Load dataset
    X, y = load_data(file_path)

    # Train models
    models_data = train_bagging_models(X, y)

    # Evaluate individual models
    print("Evaluating Individual Models:")
    evaluate_individual_models(models_data)

    # Evaluate the ensemble
    evaluate_ensemble(models_data)

    # Launch GUI
    create_gui(models_data)


if __name__ == "__main__":
    main()
