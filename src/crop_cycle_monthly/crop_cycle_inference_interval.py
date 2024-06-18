import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import pickle
import argparse
import subprocess
from tqdm import tqdm
import numpy as np  
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry
from tqdm import tqdm
from tensorflow.keras.models import load_model
from shapely.wkt import loads
from rasterio.mask import mask

# from tensorflow.keras.models import load_model

from config.config import Config
from utils.postgres_utils import *
from utils.raster_utils import *

import warnings
warnings.filterwarnings("ignore")

## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

## Modify annotations path here
ANNOTATIONS_PATH = os.path.join(CURRENT_DIR, "annotations", "crop_data_after_final.csv")
MODEL_PATH = os.path.join(CURRENT_DIR, "weights", "crop_cycle_predictor_LSTM.keras")

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Compute crop presence for villages and add to database') 
    parser.add_argument("-i", "--interval", type=str, help="Interval type: fortnightly, monthly, weekly",default='monthly')  
    parser.add_argument('-y', '--year', type=int, help='Year', default=None)
    args = parser.parse_args()


    config = Config()
    pgconn_obj = PGConn(config)
    pgconn=pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]

    if args.year is None:
        args.year = config.setup_details["months"]["agricultural_months"][0][0]

    if not check_column_exists(pgconn_obj, table["schema"], table["table"], f"crop_cycle_{args.interval}_{args.year}_{args.year+1}"):
        sql_query = f"""
        alter table
            {table["schema"]}.{table["table"]}
        add column
            crop_cycle_{args.interval}_{args.year}_{args.year+1} text        
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)

    months_data = ['01','02','03','04','05','06',
                   '07','08','09','10','11','12']
    months_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    cycle_months = config.setup_details["months"]["agricultural_months"]
    columns = ""
    for i in range(len(cycle_months)):
        # columns += f"{months_names[month[1]]}_{month[0]}_crop_presence,"
        columns += f"crop_presence_{args.year}_monthly_{i+1},"
    
    ## Tensorflow model
    if MODEL_PATH.endswith((".keras", ".h5")):
        model = load_model(MODEL_PATH)

    ## Scikit-learn model
    else:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)  
            print(f"Model : {model}")
            # crop_cycle_map = {i: class_ for i,class_ in  enumerate(model.classes_)}


    crop_cycle_map = {
        0: "kharif_rabi",
        1: "long_kharif",
        2: "no_crop",
        3: "perennial",
        4: "short_kharif",
        5: "weed",
        6: "zaid"
    }

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

    for key in tqdm(keys, desc="Processing farmplots"):

        sql_query = f"""
        select
            {columns}
            {table["key"]}
        from
            {table["schema"]}.{table["table"]}
        where
            {table["key"]} = {key}
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)
            row = curs.fetchall()[0]
        
        crop_presence_vec = np.array([[float(item) for item in row[0:-1]],])
        if "svc" in MODEL_PATH.lower():
            crop_cycle = crop_cycle_map[model.predict(crop_presence_vec)[0]]
            
        elif "keras" in MODEL_PATH.lower():
            crop_presence_vec = np.array([[float(item) for item in row[0:-1]]])
            crop_presence_vec = np.expand_dims(crop_presence_vec, axis=1)
            crop_cycle_bits = model.predict(crop_presence_vec)
            crop_cycle = crop_cycle_map[np.argmax(crop_cycle_bits)]

        else:
            crop_cycle_bits = model.predict_proba(crop_presence_vec)
            crop_cycle = crop_cycle_map[np.argmax(crop_cycle_bits)]
  
        sql_query = f"""
        update
            {table["schema"]}.{table["table"]}
        set
            crop_cycle_{args.interval}_{args.year}_{args.year+1} = '{crop_cycle}'
        where
            {table["key"]} = {key}
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)

    print("\n\n Done !")