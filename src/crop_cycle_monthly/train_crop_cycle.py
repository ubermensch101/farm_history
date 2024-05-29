import os
import argparse
import subprocess
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

## Modify annotations path here
ANNOTATIONS_PATH = os.path.join(CURRENT_DIR, "annotations", "crop_data_after_final.csv")
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "weights")

lstm = Sequential([
            LSTM(32, activation='elu', input_shape=(1, 12)),  
            Dense(7, activation='softmax') 
        ])

lstm.compile(optimizer=AdamW(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_dict={
    "SVC": SVC(),
    "LR": LogisticRegression(),
    "GB": GradientBoostingClassifier(),
    "RF": RandomForestClassifier(),
    "LSTM": lstm
}

model2text = {
    "SVC": "Support Vector Machine",
    "RF" : "Random Forest", 
    "GB" : "Gradient Boosting"    
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="SVC")
    args = parser.parse_args()

    model = model_dict[args.model.upper()]
    print(f"Model : {model}\n")
    
    # Load data
    X = []
    Y = []
    with open(ANNOTATIONS_PATH) as file:
        for line in file.readlines():
            comma_split = line.split(",")
            X_Y = [item.strip() for item in comma_split]
            Y.append(X_Y[-1])
            X.append([float(item) for item in X_Y[0:-1]])
    X = np.array([np.array(item) for item in X])

    # crop_cycle_map = {
    #     "kharif_rabi": np.array([1, 0, 0, 0, 0, 0, 0]),
    #     "no_crop": np.array([0, 1, 0, 0, 0, 0, 0]),
    #     "short_kharif": np.array([0, 0, 1, 0, 0, 0, 0]),
    #     "long_kharif": np.array([0, 0, 0, 1, 0, 0, 0]),
    #     "zaid": np.array([0, 0, 0, 0, 1, 0, 0]),
    #     "perennial": np.array([0, 0, 0, 0, 0, 1, 0]),
    #     "weed": np.array([0, 0, 0, 0, 0, 0, 1]),
    # }

    crop_cycle_map = {
        0: "kharif_rabi",
        1: "long_kharif",
        2: "no_crop",
        3: "perennial",
        4: "short_kharif",
        5: "weed",
        6: "zaid"
    }

    # Y = [crop_cycle_inv_map[item] for item in Y]
    Y = LabelEncoder().fit_transform(Y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )

    ## Fit Model
    if args.model.upper() == "LSTM":
        X_train = np.round(X_train, decimals=1)
        X_test = np.round(X_test, decimals=1)
        def discretize_probabilities(X):
            intervals = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            discretized_X = np.digitize(X, intervals, right=True) - 1
            return discretized_X

        # Discretize the rounded probabilities
        X_train = discretize_probabilities(X_train)
        X_test = discretize_probabilities(X_test)
        X_train_lstm = np.expand_dims(X_train, axis=1)
        X_test_lstm = np.expand_dims(X_test, axis=1)
    
        ## Train Model
        history = model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=1)
       
        y_train_pred = np.argmax(model.predict(X_train_lstm), axis=1)
        y_test_pred = np.argmax(model.predict(X_test_lstm),axis=1)

        print(y_test_pred, y_train_pred)
        ## Evaluate model
        train_acc = accuracy_score(y_train,y_train_pred )
        test_acc = accuracy_score(y_test, y_test_pred)
        results = f"Train Accuracy: {train_acc*100:.2f}%\nTest Accuracy: {test_acc*100:.2f}%"

        print(results)

        "Train - Report\n"
        print(classification_report(y_train, y_train_pred))
        "Test- Report\n"
        print(classification_report(y_test, y_test_pred))
   
   
   
   
    else:
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        ## Evaluate model
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        results = f"Train Accuracy: {train_acc}\nTest Accuracy: {test_acc}"

        print(results)

        "Train - Report\n"
        print(classification_report(y_train, y_train_pred))
        "Test- Report\n"
        print(classification_report(y_test, y_test_pred))

    # print("Confusion Matrix Train")
    # print(confusion_matrix(y_train, y_train_pred))


    # print("\n\nConfusion Matrix Test")
    # print(confusion_matrix(y_test, y_test_pred))

    if args.model.upper() == "LSTM":
        MODEL_PATH = os.path.join(CHECKPOINT_PATH, f"crop_cycle_predictor_LSTM.keras")
        model.save(MODEL_PATH)
    else:
        MODEL_PATH = os.path.join(CHECKPOINT_PATH, f"crop_cycle_predictor_{args.model.upper()}.pkl")
        with open(os.path.join(CHECKPOINT_PATH, f"crop_cycle_predictor_{args.model.upper()}.pkl"),"wb") as file:
            pickle.dump(model, file)

    print("Saving Weigths to :",MODEL_PATH)
    




