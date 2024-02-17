import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

X = []
Y = []
with open('cycle_annotations.csv') as file:
    for line in file.readlines():
        comma_split = line.split(',')
        X_Y = [item.strip() for item in comma_split]
        Y.append(X_Y[-1])
        X.append([float(item) for item in X_Y[0:-1]])

print(X)
print(Y)
crop_cycle_map = {
    "kharif_and_rabi": (0,0,1,0),
    "short_kharif": (1,0,0,0),
    "long_kharif": (0,1,0,0),
    "perennial": (0,0,0,1)
}
Y = [crop_cycle_map[item] for item in Y]
print(Y)

# Building the model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(12,)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
model.fit(X, Y, epochs=100, batch_size=32)

model.save('crop_cycle_predictor')