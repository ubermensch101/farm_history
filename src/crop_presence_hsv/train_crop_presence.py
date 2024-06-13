import numpy as np
import pickle
import os
import argparse
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from config.config import Config
from utils.postgres_utils import PGConn
from utils.raster_utils import *
from utils.report_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "crop_presence")
CHECKPOINTS_DIR = os.path.join(CURRENT_DIR, "weights")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-type", type=str, help="Model type", default="LR")
    parser.add_argument("-p","--data-path", type=str, default=None, help="Path to training data")
    parser.add_argument("--no-report", action='store_true', help="Do not generate the training report")
    args = parser.parse_args()

    if args.data_path is None:
        print("Using default data path")
    models = {
        "LR": LogisticRegression(),
        "RF": RandomForestClassifier(),
        "GB": GradientBoostingClassifier()
    }
    model2text = {
        "LR" : "Logistic Regression",
        "RF" : "Random Forest", 
        "GB" : "Gradient Boosting"    
    }

    # Getting annotations
    X = []
    Y = []
    yes_dir = os.path.join(DATA_DIR, "y")
    farmplots = os.listdir(yes_dir)
    for farmplot in farmplots:
        X.append(compute_hue_features(os.path.join(yes_dir, farmplot)))
        Y.append(1)
    no_dir = os.path.join(DATA_DIR, "n")
    farmplots = os.listdir(no_dir)
    for farmplot in farmplots:
        X.append(compute_hue_features(os.path.join(no_dir, farmplot)))
        Y.append(0)

    ## Define data split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    ## Train Model
    model = models[args.model_type.upper()]
    model.fit(X_train, Y_train)

    predictions_test = model.predict(X_test)
    predictions_train = model.predict(X_train)
    
    # Save the model
    with open(os.path.join(CHECKPOINTS_DIR, f"crop_presence_detector_{args.model_type.upper()}.pkl"), 'wb') as file:
        pickle.dump(model, file)

    ## Generate model report
    pdf_path = os.path.join(CURRENT_DIR, f"report_{args.model_type.upper()}.pdf")
    generate_model_report(model, X_train, Y_train, X_test,Y_test, output=pdf_path)