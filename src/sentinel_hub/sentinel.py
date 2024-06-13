"""
Desciption:
    Fetches satellite data from Sentinel Hub API for a given interval and year.
    
    Input:
        - year: Agricultural year to be processed
        - interval: Interval for fetching data (weekly, fortnightly, monthly)

    Output:
        - Satellite data in TIFF format for the given interval and year

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
    length = end - start
    print("length:", length)
    interval_type = interval_type.lower()
    if interval_type == 'weekly':
        delta = length/48
    elif interval_type == 'fortnightly':
        delta = length/24
    elif interval_type == 'monthly':
        delta = length/12  # Rough approximation of a month
    else:
        raise ValueError("Invalid interval type. Choose from 'weekly', 'fortnightly', or 'monthly'.")

    print("delta:", delta)
    intervals = []
    current_start = start
    i = 1
    while current_start < end:
        current_end = current_start + delta
        if current_end > end:
            current_end = end
        intervals.append((current_start, current_end, i))
        current_start = current_end
        i += 1

    return intervals

if __name__=='__main__':

    ## Sentinel API Configuration
    sh_config = SHConfig()
    sh_config.sh_client_id = '4796405e-1d2a-4b46-9813-bc8d58cc1d3f'
    sh_config.sh_client_secret = 'MGpYNqJcsqqFah9jNnhM0lDp0KvKUkyE' 

    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", type=int, default= None,
                        help="Agricultural year to be processed")
    parser.add_argument("-i", "--interval", type=str, choices=["weekly", "fortnightly", "monthly"], default="monthly",
                        help="Interval for fetching data (weekly, fortnightly, monthly)")
    
    args = parser.parse_args()
    year = args.year
    interval_type = args.interval

    # months_data = ['01','02','03','04','05','06',
    #             '07','08','09','10','11','12']

    months_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    config = Config()
    pgconn_obj = PGConn(config)
    pgconn=pgconn_obj.connection()

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

    ## If year argument is provided, fetch data for that year only
    if year is not None:
        start_year = year
        end_year = year+1
        start_month = 5
        end_month = 4
        year = start_year

    ## Else fetch data for the agricultural year/years present in config/months.json
    else:
        start_year = config.setup_details['months']['agricultural_months'][0][0]
        start_month = config.setup_details['months']['agricultural_months'][0][1] + 1
        end_year = config.setup_details['months']['agricultural_months'][-1][0]
        end_month = config.setup_details['months']['agricultural_months'][-1][1] + 1
      
    start = datetime.date(start_year, start_month, 1)           
    end = datetime.date(end_year, end_month, 30)
    length = end - start
    intervals = get_intervals(start, end, interval_type)
    
    for interval in intervals:
        
        data_folder = os.path.join(ROOT_DIR, interval_type, table["table"], f"{year}", f"{interval[2]}")       
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
