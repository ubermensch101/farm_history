import argparse
import datetime
import os
import subprocess
import json
import shutil

from sentinelhub import (
    CRS,
    DataCollection,
    Geometry,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    MosaickingOrder,
    SentinelHubInputTask,
)
y = SentinelHubInputTask(DataCollection.SENTINEL1_IW, time_interval=('2021-01-01', '2021-01-31'), mosaicking_order=MosaickingOrder.LEAST_CC)

from config.config import Config
from utils.postgres_utils import PGConn

ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())

def get_weekly_intervals(year):
    start = datetime.date(year, 5, 1)  # Start from May 1st of the specified year
    end = datetime.date(year + 1, 4, 30)  # End on April 30th of the following year

    intervals = []
    current_date = start
    while current_date <= end:
        intervals.append((current_date, current_date + datetime.timedelta(days=6)))
        current_date += datetime.timedelta(days=7)  # Increment by 7 days (weekly interval)

    return intervals


def format_folder_name(start_date):
    return start_date.strftime("%B_%d_%Y")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", type=int, required=True,
                        help="Agricultural year to be processed")
    args = parser.parse_args()
    year = args.year

    sh_config = SHConfig()
    sh_config.sh_client_id = 'd5fd0f7c-3a77-4c94-9984-86f66cdaf1ac'
    sh_config.sh_client_secret = 'IBSSLtzgdR8PPgHBsIvRNaL4FrMM0qbF'

    config = Config()
    pgconn_obj = PGConn(config)
    pgconn = pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]

    sql_query = f"""
    SELECT
        st_asgeojson(st_envelope(st_union(st_transform({table["geom_col"]}, 4674))))
    FROM
        {table["schema"]}.{table["table"]}
    LIMIT
        1
    ;
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        polygon = json.loads(curs.fetchall()[0][0])
        full_geometry = Geometry(polygon, crs=CRS.WGS84)

    # Function to download Sentinel data
    def download_sentinel_data(data_collection, evalscript, data_folder, time_interval):
        request = SentinelHubRequest(
            evalscript=evalscript,
            data_folder=data_folder,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=data_collection,
                    time_interval=time_interval
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            geometry=full_geometry,
            config=sh_config,
            resolution=(10,10)
        )

        try:
            request.get_data(save_data=True, show_progress=True)
        except Exception as e:
            print(f"Failed to download {data_collection} data: {e}")

    # Evalscript for Sentinel-1 RGB visualization
    evalscript_sentinel1_rgb = """
    //VERSION=3
    function setup() {
        return {
            input: ["VV", "VH"],
            output: { bands: 3 }
        };
    }

    function evaluatePixel(sample) {
        let VV = sample.VV;
        let VH = sample.VH;
        
        // Calculate ratio VV/VH, handle zero VH case
        let ratio = (VV !== 0) ? VH / VV : 1.0; // Fallback to 1.0 if VH is zero
        
        // Create RGB composite
        return [VV, VH, ratio];
    }
    """

    # Generate weekly intervals
    intervals = get_weekly_intervals(year)

    parent_folder_name = f"Sentinel_Data_Normalized_{year}"

    for interval in intervals:
        start_date, end_date = interval
        folder_name = format_folder_name(start_date)
        output_folder = os.path.join(ROOT_DIR, parent_folder_name, folder_name)
        data_folder_sentinel1_rgb = os.path.join(output_folder, "sentinel1_rgb")

        # Create output folder if it doesn't exist
        os.makedirs(data_folder_sentinel1_rgb, exist_ok=True)

        # Download Sentinel-1 RGB data
        download_sentinel_data(DataCollection.SENTINEL1_IW, evalscript_sentinel1_rgb, data_folder_sentinel1_rgb, 
                               (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))

        print(f"Downloaded data for {folder_name}")

    print("All data downloaded successfully.")