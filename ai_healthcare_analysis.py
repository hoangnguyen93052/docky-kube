import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    print(f"Data loaded with shape: {data.shape}")
    return data

# Preprocess the data
def preprocess_data(data):
    # Handle missing values
    data.dropna(inplace=True)  # For simplicity, drop missing values
    print(f"Data shape after dropping missing values: {data.shape}")
    
    # Encode categorical variables
    data = pd.get_dummies(data, drop_first=True)
    print(f"Data shape after encoding: {data.shape}")
    
    return data

# Split the dataset into features and target
def split_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y

# Scale the features
def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled")
    return X_scaled, scaler

# Train a RandomForest model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained")
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Save the model
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

# Load the model
def load_model(filename):
    model = joblib.load(filename)
    print("Model loaded")
    return model

# Main function
def main():
    filepath = 'healthcare_data.csv'  # Change this to the path of your dataset
    target_column = 'target'  # Change this to your actual target column name

    # Load data
    data = load_data(filepath)

    # Preprocess data
    data = preprocess_data(data)

    # Split data
    X, y = split_data(data, target_column)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled, _ = scale_features(X_test)

    # Train model
    model = train_model(X_train_scaled, y_train)

    # Evaluate model
    evaluate_model(model, X_test_scaled, y_test)

    # Save the model
    save_model(model, 'healthcare_model.joblib')

if __name__ == "__main__":
    main()