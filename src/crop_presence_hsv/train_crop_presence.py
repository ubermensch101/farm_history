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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
DATA_DIR = os.path.join(ROOT_DIR, "data", "crop_presence")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-type", type=str, help="Model type", default="LR")
    parser.add_argument("--no-report", action='store_true', help="Do not generate the training report")
    args = parser.parse_args()

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

    model = models[args.model_type.upper()]
    model.fit(X_train, Y_train)

    predictions_test = model.predict(X_test)
    predictions_train = model.predict(X_train)

    if not args.no_report:
        # Classification Report
        class_report = classification_report(Y_test, predictions_test, output_dict=True)
        class_report_text = classification_report(Y_test, predictions_test)

        # Confusion Matrix
        conf_matrix_test = confusion_matrix(Y_test, predictions_test)
        conf_matrix_train = confusion_matrix(Y_train, predictions_train)

        # ROC Curve
        fpr, tpr, _ = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        # Generate Plots
        sns.set(style='whitegrid')
        pdf_path = os.path.join(CURRENT_DIR, f"report_{args.model_type.upper()}.pdf")
        with PdfPages(pdf_path) as pdf:
            # Page 1: Title, Text Report, and ROC Curve
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69))

            # Title and Text Report
            ax1.text(0.5, 1.08, "Training Report", fontsize=24, ha='center', va='top', transform=ax1.transAxes)
            ax1.text(0.5, 0.9, f"Crop Presence Detector - {model2text[args.model_type.upper()]}", fontsize=18, ha='center',va='top', transform=ax1.transAxes)
            ax1.text(0.01, 0.5, str(class_report_text), {'fontsize': 12}, fontproperties='monospace', va='top', transform=ax1.transAxes)
            ax1.axis('off')

            # ROC Curve

            ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax2.legend(loc="lower right")

            fig.tight_layout(pad=6.0)
            pdf.savefig(fig)
            plt.close()

            # Page 2: Confusion Matrix - Test Set and Train Set
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69))

            sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title('Confusion Matrix - Test Set')

            sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', ax=ax2)
            ax2.set_title('Confusion Matrix - Train Set')

            fig.tight_layout(pad=6.0)
            pdf.savefig(fig)
            plt.close()

    # Save the model
    with open(os.path.join(CHECKPOINTS_DIR, f"crop_presence_detector_{args.model_type.upper()}.pkl"), 'wb') as file:
        pickle.dump(model, file)

    probs = model.predict_proba(X_test)[:, 1]
    # Number of bootstrap iterations
    n_iterations = 1000

    # Function to calculate confidence intervals
    def calculate_confidence_intervals(model, X, y, n_iterations):
        probabilities = []
        for _ in range(n_iterations):
            # Bootstrap sample
            X_boot, y_boot = resample(X, y)
            # Fit model on bootstrap sample
            model.fit(X_boot, y_boot)
            # Predict probabilities for original data 
            probabilities.append(model.predict_proba(X)[:, 1])
        
        # Calculate confidence intervals
        lower_bound = np.percentile(probabilities, 2.5, axis=0)
        upper_bound = np.percentile(probabilities, 97.5, axis=0)
        return lower_bound, upper_bound

    # Calculate confidence intervals
    lower_bound, upper_bound = calculate_confidence_intervals(model, X_test, Y_test, n_iterations)

    for i in range(len(lower_bound)):
        print(f"Farmplot {i}: Prediction {probs[i]:.2f} \t {lower_bound[i]:.2f} - {upper_bound[i]:.2f}")