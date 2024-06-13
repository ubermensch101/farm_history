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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", type=int, required=True,
                        help="Agricultural year to be processed")
    args = parser.parse_args()
    year = args.year

    sh_config = SHConfig()
    sh_config.sh_client_id = '4796405e-1d2a-4b46-9813-bc8d58cc1d3f'
    sh_config.sh_client_secret = 'MGpYNqJcsqqFah9jNnhM0lDp0KvKUkyE'                

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

    start = datetime.date(year, 6, 1)
    end = datetime.date(year + 1, 5, 31)
    length = end - start
    print("length:", length)
    interval = length/24
    print("interval:", interval)
    fortnights = []
    for i in range(24):
        fortnight_start = start + i*interval
        fortnight_end = start + (i + 1)*interval
        fortnights.append((fortnight_start, fortnight_end, i+1))

    for fortnight in fortnights:
        data_folder = os.path.join(ROOT_DIR, "fortnightly", table["table"] + "_" + str(year), str(fortnight[2]))
        request = SentinelHubRequest(
            data_folder=data_folder,
            evalscript=evalscript_true_color,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(fortnight[0].strftime("%Y-%m-%d"),
                                   fortnight[1].strftime("%Y-%m-%d")),
                    mosaicking_order=MosaickingOrder.LEAST_CC
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            geometry=full_geometry,
            # size=(2048,2048),
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
