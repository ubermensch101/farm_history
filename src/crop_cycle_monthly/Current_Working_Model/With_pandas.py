import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv("/home/rahul/farm_history/src/crop_cycle_monthly/crop_data_after_final.csv", header=None)

# Extract features (X) and labels (Y)
X = df.iloc[:, 0:-1].values  # Exclude the first and last columns
Y = df.iloc[:, -1].values  # Last column as labels

# Define the crop cycle mapping
crop_cycle_map = {
    "kharif_rabi": np.array([1, 0, 0, 0, 0, 0, 0]),
    "no_crop": np.array([0, 1, 0, 0, 0, 0, 0]),
    "short_kharif": np.array([0, 0, 1, 0, 0, 0, 0]),
    "long_kharif": np.array([0, 0, 0, 1, 0, 0, 0]),
    "zaid": np.array([0, 0, 0, 0, 1, 0, 0]),
    "perennial": np.array([0, 0, 0, 0, 0, 1, 0]),
    "weed": np.array([0, 0, 0, 0, 0, 0, 1]),
}

# Map labels to numerical vectors
#Y_mapped = np.array([crop_cycle_map[item] for item in Y])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.4, random_state=42
)

# Train the Support Vector Classifier
model = SVC()
model.fit(X_train, y_train)

# Evaluate the model
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
results = f"Train Accuracy: {train_acc}\nTest Accuracy: {test_acc}"
print(results)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
