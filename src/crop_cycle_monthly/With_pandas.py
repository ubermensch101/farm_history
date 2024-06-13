import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv("/home/rahul/farm_history/src/crop_cycle_monthly/crop_data_after_final.csv", header=None)

# Extract features (X) and labels (Y)
X = df.iloc[:, 0:-1].values  # Exclude the last column
Y = df.iloc[:, -1].values  # Last column as labels

# Encode labels to integers
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y_encoded, test_size=0.2, random_state=82
)

# Reshape input data for LSTM (add a new axis for time steps)
X_train_lstm = np.expand_dims(X_train, axis=1)
X_test_lstm = np.expand_dims(X_test, axis=1)

# Define the Recurrent Neural Network (RNN) model using LSTM
model = Sequential([
    LSTM(32, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),  
    Dense(7, activation='softmax') 
])

# Compile the model with Adam optimizer
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
history = model.fit(X_train_lstm, y_train, epochs=200, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=1)

# Save the model
model.save("/home/rahul/farm_history/src/crop_cycle_monthly/lstm_model.keras")

# Get the probabilities and predicted classes for test data
y_pred_prob = model.predict(X_test_lstm)
y_pred_class = np.argmax(y_pred_prob, axis=1)
print(y_pred_prob)
print('+'*100)
# Calculate confidence values (maximum probability for each prediction)

confidence_values = np.max(y_pred_prob, axis=1)

# Print confidence values
print("Confidence values for each prediction:")
print(confidence_values)
print(len(confidence_values))

# Plotting training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()
