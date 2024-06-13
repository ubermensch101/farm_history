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
from sklearn.preprocessing import label_binarize
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
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))



def generate_model_report(model, X_train, Y_train, X_test, Y_test, output: str):
    predictions_test = model.predict(X_test)
    predictions_train = model.predict(X_train)
    class_report = classification_report(Y_test, predictions_test, output_dict=True)
    class_report_text = classification_report(Y_test, predictions_test)

    # Confusion Matrix
    conf_matrix_test = confusion_matrix(Y_test, predictions_test)
    conf_matrix_train = confusion_matrix(Y_train, predictions_train)

    sns.set_theme(style='whitegrid')

    with PdfPages(output) as pdf:
        # Page 1: Title
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.text(0.5, 0.5, "Training Report\nCrop Presence Detector", fontsize=24, ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        fig.tight_layout(pad=6.0)
        pdf.savefig(fig)
        plt.close()

        # Page 2: ROC Curve and Text Report
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69))

        # ROC Curve
        if len(np.unique(Y_test)) > 2:  # Multi-class case
            Y_test_bin = label_binarize(Y_test, classes=np.unique(Y_test))
            n_classes = Y_test_bin.shape[1]
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], model.predict_proba(X_test)[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            for i, color in zip(range(n_classes), sns.color_palette("hsv", n_classes)):
                ax1.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve class {i} (area = {roc_auc[i]:0.2f})')
            ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        else:  # Binary case
            fpr, tpr, _ = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
            ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax1.legend(loc="lower right")

        # Text Report
        ax2.text(0.01, 1.0, str(class_report_text), {'fontsize': 12}, fontproperties='monospace', va='top', ha='left', transform=ax2.transAxes)
        ax2.axis('off')

        fig.tight_layout(pad=6.0)
        pdf.savefig(fig)
        plt.close()

        # Page 3: Confusion Matrix - Test Set and Train Set
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69))

        sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix - Test Set')
        ax1.set_xlabel('Predicted Labels')
        ax1.set_ylabel('True Labels')

        sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('Confusion Matrix - Train Set')
        ax2.set_xlabel('Predicted Labels')
        ax2.set_ylabel('True Labels')

        fig.tight_layout(pad=6.0)
        pdf.savefig(fig)
        plt.close()
# def generate_model_report(model, X_train, Y_train, X_test, Y_test, output:str):  
#     predictions_test = model.predict(X_test)
#     predictions_train = model.predict(X_train)
#     class_report = classification_report(Y_test, predictions_test, output_dict=True)
#     class_report_text = classification_report(Y_test, predictions_test)

#     # Confusion Matrix
#     conf_matrix_test = confusion_matrix(Y_test, predictions_test)
#     conf_matrix_train = confusion_matrix(Y_train, predictions_train)

#     # ROC Curve
#     fpr, tpr, _ = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])
#     roc_auc = auc(fpr, tpr)

#     # Generate Plots
#     sns.set_theme(style='whitegrid')

#     with PdfPages(output) as pdf:
#         # Page 1: Title, Text Report, and ROC Curve
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69))

#         # Title and Text Report
#         ax1.text(0.5, 1.08, "Training Report", fontsize=24, ha='center', va='top', transform=ax1.transAxes)
#         ax1.text(0.5, 0.9, f"Crop Presence Detector", fontsize=18, ha='center',va='top', transform=ax1.transAxes)
#         ax1.text(0.01, 0.5, str(class_report_text), {'fontsize': 12}, fontproperties='monospace', va='top', transform=ax1.transAxes)
#         ax1.axis('off')

#         # ROC Curve

#         ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
#         ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         ax2.set_xlim([0.0, 1.0])
#         ax2.set_ylim([0.0, 1.05])
#         ax2.set_xlabel('False Positive Rate')
#         ax2.set_ylabel('True Positive Rate')
#         ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
#         ax2.legend(loc="lower right")

#         fig.tight_layout(pad=6.0)
#         pdf.savefig(fig)
#         plt.close()

#         # Page 2: Confusion Matrix - Test Set and Train Set
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69))

#         sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', ax=ax1)
#         ax1.set_title('Confusion Matrix - Test Set')

#         sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', ax=ax2)
#         ax2.set_title('Confusion Matrix - Train Set')

#         fig.tight_layout(pad=6.0)
#         pdf.savefig(fig)
#         plt.close()



