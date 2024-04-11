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
from models import *

from config.config import *
from utils.postgres_utils import *
from utils.raster_utils import *


## TRAINING CONFIG
dir_path = os.path.dirname(__file__)
setup_file = os.path.join(dir_path,"train_config.json")
with open(setup_file,'r') as file:
    train_config = json.loads(file.read())

## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
RASTER_PATH = os.path.join(ROOT_DIR, "src", "crop_presence_cnn","temp_clipped.tif")


# Pre-processing    
model_types = ["hist", "hsv", "rgb"]

model_transforms = {
    "hist": transforms.Compose([
                transforms.ToTensor(), 
                ToHistogram(bins=train_config['bins'])
                ]),
                   
    "rgb": transforms.Compose([
                transforms.ToTensor(),
                ]),
    "hsv": transforms.Compose([
                ToHSV(),
                transforms.ToTensor()
                ])
}

## Utilities
def load_model(model_type):
    if model_type.lower() in ["rgb", "hsv"]:
        model= SimpleCNN()
        model.load(train_config[f'checkpoint_path_{model_type.lower()}'])

    elif model_type.lower() in ["hist"]:
        model = torch.load(train_config[f'checkpoint_path_{model_type.lower()}'])

    return model



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-type", type=str, required=True, help= "hist -> histogram Model \n rgb -> CNN for RGB, \n hsv-> CNN for HSV")
    args = parser.parse_args()

    if args.model_type.lower() not in model_types:
        raise ValueError("Model doesn't exist")

    # Load crop prediciton model and configurations
    transform = model_transforms[args.model_type.lower()]
    model = load_model(args.model_type)

    ## Load Database config
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
                               f'{months_names[month[1]]}_{month[0]}_crop_presence_ml'):
            sql_query = f"""
            alter table
                {table["schema"]}.{table["table"]}
            add column
                {months_names[month[1]]}_{month[0]}_crop_presence_ml numeric
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
        print(f"Running Crop Detection on farm ID: {farm[0]}")
        for month in ml_months:

            multipolygon = loads(farm[1])
            super_clip('quads', month[0], months_data[month[1]], multipolygon, RASTER_PATH)
            raw_poly = farm[2]
            polygon = [(float(item.split(' ')[0]), float(item.split(' ')[1])) for item in raw_poly.strip().split('(')[3].split(')')[0].split(',')]
            remove_padding(RASTER_PATH, RASTER_PATH)

            # image = np.array(Image.open(RASTER_PATH))[:,:,:3]       ## Drop Near IR regions 
            # input_tensor = transform(image).unsqueeze(0)
            image  = Image.fromarray(np.array(Image.open(RASTER_PATH))[:,:,:3])
            input_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                # Apply softmax to get probabilities
                probabilities = F.softmax(output, dim=1).flatten()

            crop_presence_prob= probabilities[1].item() 

            sql_query = f"""
            update
                {table["schema"]}.{table["table"]}
            set
                {months_names[month[1]]}_{month[0]}_crop_presence_ml = {crop_presence_prob}
            where
                {table["key"]} = {farm[0]}
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)
    pgconn.commit()



  










    


