import os
import argparse
import torch
import subprocess
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from shapely.wkt import loads
from train_utils import *
from train_utils import *


from config.config import *
from utils.postgres_utils import *
from utils.raster_utils import *

## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
RASTER_PATH = os.path.join(ROOT_DIR, "src", "crop_presence_cnn","temp_clipped.tif")
DEFAULT_CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights", "crop_presence_hist.pt")


## DATABASE CONFIG
dir_path = os.path.dirname(__file__)
setup_file = os.path.join(dir_path,"train_config.json")
with open(setup_file,'r') as file:
    train_config = json.loads(file.read())


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH)
    args = parser.parse_args()


    # Pre-processing    
    transform_hsv = transforms.Compose([
    transforms.ToTensor(), 
    ToHistogram(bins=train_config['bins']),  # Calculate normalized histograms for each channel
    ])
    # Load crop prediciton model 
    model = torch.load(DEFAULT_CHECKPOINT_PATH)

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

    sql_query = f"""
    select
        {table["key"]},
        st_astext(st_multi(st_buffer(st_transform({table["geom_col"]}, 3857), 50))),
        st_astext(st_multi(st_transform({table["geom_col"]}, 3857)))
    from
        {table["schema"]}.{table["table"]}
    where
        {table["filter"]}
    order by
        {table["key"]}

    """

    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        poly_fetch_all = curs.fetchall()


    for farm in poly_fetch_all:
        for month in ml_months:

            multipolygon = loads(farm[1])
            super_clip('quads', month[0], months_data[month[1]], multipolygon, RASTER_PATH)
            raw_poly = farm[2]
            polygon = [(float(item.split(' ')[0]), float(item.split(' ')[1])) for item in raw_poly.strip().split('(')[3].split(')')[0].split(',')]
            remove_padding(RASTER_PATH, RASTER_PATH)
            #f'{table["key"]}: {farm[0]}, {months_names[month[1]]}-{month[0]}'
            # Load image
            image = Image.open(RASTER_PATH)
            image= np.array(image)

            image = image[:,:,:3]   ## Drop Near IR regions
            input_tensor = transform_hsv(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                # Apply softmax to get probabilities
                probabilities = F.softmax(output, dim=1).flatten()

            crop_presence = probabilities[1].item() 


            sql_query = f"""
            update
                {table["schema"]}.{table["table"]}
            set
                {months_names[month[1]]}_{month[0]}_crop_presence = {crop_presence}
            where
                {table["key"]} = {farm[0]}
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)

    pgconn.commit()



  










    


