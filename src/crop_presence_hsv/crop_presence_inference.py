import numpy as np
import argparse
import pickle
import subprocess
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from config.config import *
from utils.postgres_utils import *

## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
DATA_DIR = os.path.join(ROOT_DIR, "data", "crop_presence")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    config = Config()
    pgconn_obj = PGConn(config)
    pgconn=pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]

    months_data = ['01','02','03','04','05','06',
                   '07','08','09','10','11','12']
    months_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    ml_months = config.setup_details["months"]["agricultural_months"]

    for month in ml_months:
        if not check_column_exists(pgconn_obj, table["schema"], table["table"],
                               f'{months_names[month[1]]}_{month[0]}_crop_presence'):
            sql_query = f"""
            alter table
                {table["schema"]}.{table["table"]}
            add column
                {months_names[month[1]]}_{month[0]}_crop_presence numeric
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)

    ## Load your model, you can change path to your model
    with open(os.path.join(CHECKPOINTS_DIR,"crop_presence_detector_new.pkl"), 'rb') as file:     
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

    for key in tqdm(keys, desc="Processing farm_id"):
        # print(f"Processing farm_id: {key}/{keys[-1]}")
        for month in ml_months:
            sql_query = f"""
            select
                {months_names[month[1]]}_{month[0]}_hue_mean,
                {months_names[month[1]]}_{month[0]}_hue_stddev
            from
                {table["schema"]}.{table["table"]}
            where
                {table["key"]} = {key}
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)
                row = curs.fetchall()[0]
            bands = [[float(item) for item in row]]
            crop_presence = model.predict_proba(bands)[0][1]
            
            sql_query = f"""
            update
                {table["schema"]}.{table["table"]}
            set
                {months_names[month[1]]}_{month[0]}_crop_presence = {crop_presence}
            where
                {table["key"]} = {key}
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)
    pgconn.commit()