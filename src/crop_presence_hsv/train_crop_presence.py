import numpy as np
import pickle
import os
import subprocess

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from config.config import Config
from utils.postgres_utils import PGConn
from utils.raster_utils import *

ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
DATA_DIR = os.path.join(ROOT_DIR, "data")

if __name__=='__main__':

    # Getting annotations
    X = []
    Y = []
    yes_dir = os.path.join(DATA_DIR, "train", "y")
    farmplots = os.listdir(yes_dir)
    for farmplot in farmplots:
        X.append(compute_hue_features(os.path.join(yes_dir, farmplot)))
        Y.append(1)
    no_dir = os.path.join(DATA_DIR, "train", "n")
    farmplots = os.listdir(no_dir)
    for farmplot in farmplots:
        X.append(compute_hue_features(os.path.join(no_dir, farmplot)))
        Y.append(0)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    print(classification_report(Y_test, predictions))
    print(confusion_matrix(Y_test, predictions))

    with open(f'{os.path.dirname(os.path.realpath(__file__))}/crop_presence_detector.pkl', 'wb') as file:
        pickle.dump(model, file)