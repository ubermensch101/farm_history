import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load data
data_file = f"{os.path.dirname(os.path.realpath(__file__))}/cycle_annotations.csv"
data = pd.read_csv(data_file)

# Extract features and labels
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

# Convert multi-labels to single-labels
crop_cycle_map = {
    "kharif_and_rabi": 0,
    "short_kharif": 1,
    "long_kharif": 2,
    "no_crop": 3,
}
Y = np.array([crop_cycle_map[item] for item in Y])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.4, random_state=42
)

# Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# Evaluate model
train_acc = accuracy_score(y_train,y_train_pred )
test_acc = accuracy_score(y_test, y_test_pred)
results = f"Train Accuracy: {train_acc}\nTest Accuracy: {test_acc}"
print(results)

print("Confusion Matrix Train")
print(confusion_matrix(y_train, y_train_pred))


print("\n\nConfusion Matrix Test")
print(confusion_matrix(y_test, y_test_pred))
# print(classification_report(y_test, predict
# ions))
# print(confusion_matrix(y_test, predictions))

# Save model
with open(
    f"{os.path.dirname(os.path.realpath(__file__))}/crop_cycle_predictor_SVC.pkl",
    "wb",
) as file:
    pickle.dump(model, file)
