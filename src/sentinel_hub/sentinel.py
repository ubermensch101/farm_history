"""
Desciption:
    Fetches satellite data from Sentinel Hub API for a given interval and year.
    
    Input:
        - year: Agricultural year to be processed
        - interval: Interval for fetching data (weekly, fortnightly, monthly)

    Output:
        - Satellite data in TIFF format for the given interval and year


        
Reference:
    https://sentinelhub-py.readthedocs.io/en/latest/sh.html
"""

import argparse
import datetime
import os
import subprocess
import json
import matplotlib.pyplot as plt
import shutil

from sentinelhub import (
    CRS,
    DataCollection,
    Geometry,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    MosaickingOrder
)

from config.config import Config
from utils.postgres_utils import PGConn

ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())

def get_intervals(start, end, interval_type):
    interval_type = interval_type.lower()
    intervals = []

    if interval_type == 'weekly':
        num_intervals = 48
        delta = (end - start) / num_intervals
    elif interval_type == 'fortnightly':
        num_intervals = 24
        delta = (end - start) / num_intervals
    elif interval_type == 'monthly':
        num_intervals = 12
        delta = (end - start) / num_intervals
    else:
        raise ValueError("Invalid interval type. Choose from 'weekly', 'fortnightly', or 'monthly'.")

    current_start = start
    for i in range(1, num_intervals + 1):
        current_end = start + i * delta
        intervals.append((current_start, current_end, i))
        current_start = current_end

    print("start :", start)
    print("end :", end)
    print("delta :", delta)
    return intervals

if __name__=='__main__':

    ## Sentinel API Configuration
    sh_config = SHConfig()
    sh_config.sh_client_id = '4796405e-1d2a-4b46-9813-bc8d58cc1d3f'
    sh_config.sh_client_secret = 'MGpYNqJcsqqFah9jNnhM0lDp0KvKUkyE' 

    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interval", type=str, choices=["weekly", "fortnightly", "monthly"], default="monthly",
                        help="Interval for fetching data (weekly, fortnightly, monthly)")

    args = parser.parse_args()
    interval_type = args.interval

    months_names = ['may', 'jun', 'jul', 'aug', 'sep', 'oct',
                     'nov', 'dec', 'jan', 'feb', 'mar', 'apr']
    
    
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn=pgconn_obj.connection()
    years = config.setup_details['months']['agricultural_years'] 
    table = config.setup_details["tables"]["villages"][0]

    
    sql_query = f"""
    select
        st_asgeojson(st_envelope(st_union(st_transform({table["geom_col"]}, 4674))))
    from
        {table["schema"]}.{table["table"]}
    limit
        1
    ;
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        polygon = json.loads(curs.fetchall()[0][0])
        full_geometry = Geometry(polygon, crs=CRS.WGS84)

    evalscript_true_color = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }
        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
    """

    for year in years:
        print(f"Fetching quads for village : {table['table'].upper()} for year {year}")
        start_year = year
        start_month = 5
        end_year = start_year+1
        end_month = 4

        start = datetime.date(start_year, start_month, 1)         
        end = datetime.date(end_year, end_month, 30)
        length = end - start
        intervals = get_intervals(start, end, interval_type)

        for interval in intervals:
            
            data_folder = os.path.join(ROOT_DIR, interval_type, table["table"], f"{year}", f"{interval[2]}")    
            if os.path.exists(data_folder):
                print("Data already exists for this interval! Skipping...")
            else:
                request = SentinelHubRequest(
                    data_folder=data_folder,
                    evalscript=evalscript_true_color,
                    input_data=[
                        SentinelHubRequest.input_data(
                            data_collection=DataCollection.SENTINEL2_L2A,
                            time_interval=(interval[0].strftime("%Y-%m-%d"),
                                        interval[1].strftime("%Y-%m-%d")),
                            mosaicking_order=MosaickingOrder.LEAST_CC
                        )
                    ],
                    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                    geometry=full_geometry,
                    config=sh_config,
                )
                request.get_data(save_data=True, show_progress=True)
                # Removing weird hash directory
                items = os.listdir(data_folder)
                weird_dir = [item for item in items if os.path.isdir(os.path.join(data_folder, item))][0]
                files_in_weird_dir = os.listdir(os.path.join(data_folder, weird_dir))
                for file in files_in_weird_dir:
                    shutil.move(os.path.join(data_folder, weird_dir, file), data_folder)
                shutil.rmtree(os.path.join(data_folder, weird_dir), ignore_errors=True)
