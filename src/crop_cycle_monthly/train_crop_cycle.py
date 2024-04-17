import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import tensorflow as tf
# from tensorflow.keras import layers, models


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


X = []
Y = []
with open(
    f"{os.path.dirname(os.path.realpath(__file__))}/cycle_annotations.csv"
) as file:
    for line in file.readlines():
        comma_split = line.split(",")
        X_Y = [item.strip() for item in comma_split]
        Y.append(X_Y[-1])
        X.append([float(item) for item in X_Y[0:-1]])
X = np.array([np.array(item) for item in X])

crop_cycle_map = {
    "kharif_and_rabi": np.array([0, 0, 1, 0]),
    "short_kharif": np.array((1, 0, 0, 0)),
    "long_kharif":  np.array((0, 1, 0, 0)),
    "no_crop": np.array((0, 0, 0, 1)),
}
Y = np.array([crop_cycle_map[item] for item in Y])


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.4, random_state=42
)


model = SVC()
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
results = f"Train Accuracy: {train_acc}\nTest Accuracy: {test_acc}"

print(results)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
# print(confusion_matrix(y_test, predictions))

with open(
    f"{os.path.dirname(os.path.realpath(__file__))}/crop_cycle_predictor_LR.pkl",
    "wb",
) as file:
    pickle.dump(model, file)



"""
Neural Network -> bad choice
"""
# # Building the model
# model = models.Sequential([
#         layers.Dense(64, activation="relu", input_shape=(12,)),
#         layers.Dropout(0.2),
#         layers.Dense(32, activation="relu"),
#         layers.Dense(4, activation="softmax"),
#     ]
# )
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# model.compile(
#     optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
# )
# # Training the model
# history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# # Plot training and validation curves
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Metric')
# plt.title('Training and Validation Curves')
# plt.legend()
# plt.show()
# model.save(f"{os.path.dirname(os.path.realpath(__file__))}/crop_cycle_predictor.keras")
