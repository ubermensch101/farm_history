"""
Description:
    This script is used to infer crop presence for each village in the database.
    The script loads the trained model from the pickle file and uses it to predict crop presence for each village.

"""

import numpy as np
import pickle
import os 
import subprocess
import argparse
from tqdm import tqdm  # Import tqdm here

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from config.config import *
from utils.postgres_utils import *

import warnings
warnings.filterwarnings("ignore")

## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CHECKPOINTS_DIR = os.path.join(CURRENT_DIR, "weights", "crop_presence_detector_LR.pkl")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute crop presence for villages and add to database') 
    parser.add_argument("-i", "--interval", type=str, help="Interval type: fortnightly, monthly, weekly", default="monthly")  
    parser.add_argument('-y', '--year', type=int, help='Year', default=None)

    args = parser.parse_args()
    args.interval = args.interval.lower()
    if args.interval == "fortnightly":
        interval_length = 24
    elif args.interval == "monthly":
        interval_length = 12
    elif args.interval == "weekly":
        interval_length = 48

    config = Config()
    pgconn_obj = PGConn(config)
    pgconn = pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]
    years = config.setup_details["months"]["agricultural_years"]
    year=years[0]
    if args.year is None:
        args.year = year
    # Add progress bar to the first for loop
    for i in tqdm(range(interval_length), desc="Adding columns"):
        if not check_column_exists(pgconn_obj, table["schema"], table["table"],
                                   f'crop_presence_{args.year}_{args.interval}_{i+1}'):
            sql_query = f"""
            alter table
                {table["schema"]}.{table["table"]}
            add column
                crop_presence_{args.year}_{args.interval}_{i+1} numeric
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)

    with open(CHECKPOINTS_DIR, 'rb') as file:
        model = pickle.load(file)

    sql_query = f"""
    select
        {table["key"]}
    from
        {table["schema"]}.{table["table"]}
    where
        {table["filter"]}
    order by
        {table["key"]}
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        rows = curs.fetchall()
    keys = [item[0] for item in rows]

    # Add progress bar to the second for loop
    for i in range(interval_length):
        print(f"# {args.interval[:-2].upper()}:", i+1)
        for key in tqdm(keys, desc="Processing farmplots"):
            sql_query = f"""
            select
                hue_mean_{args.year}_{args.interval}_{i+1},
                hue_stddev_{args.year}_{args.interval}_{i+1},
                ir_mean_{args.year}_{args.interval}_{i+1},
                ir_stddev_{args.year}_{args.interval}_{i+1}
            from
                {table["schema"]}.{table["table"]}
            where
                {table["key"]} = {key}
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)
                row = curs.fetchall()[0]
            bands = [[float(item) for item in row]]
            if bands[0][0]==bands[0][1]==-1:
                crop_presence = -1
            else:
                crop_presence = model.predict_proba(bands)[0][1]
            
            sql_query = f"""
            update
                {table["schema"]}.{table["table"]}
            set
                crop_presence_{args.year}_{args.interval}_{i+1} = {crop_presence}
            where
                {table["key"]} = {key}
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)
    pgconn.commit()