import argparse
import csv
import matplotlib.pyplot as plt
from shapely.wkt import loads
import pandas as pd
from config.config import Config
from utils.postgres_utils import *
from utils.raster_utils import *

if __name__=='__main__':
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn=pgconn_obj.connection()

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_key", type=int, required=True)
    parser.add_argument("-e", "--end_key", type=int, required=True)
    args = parser.parse_args()

    table = {
        "schema": "pilot", 
        "table": "bid_558923",
        "geom_col": "wkb_geometry",
        "key": "ogc_fid"
    }

    months_data = ['01','02','03','04','05','06',
                   '07','08','09','10','11','12']
    months_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    annotate_months = config.setup_details["months"]["agricultural_months"]
    columns = [{table["key"]}]
    for month in annotate_months:
        columns.append(f"{months_names[month[1]]}_{month[0]}_crop_presence")

    df = pd.DataFrame(columns=columns)

    sql_query = f"""
    select
        {table["key"]},
        st_astext(st_multi(st_buffer(st_transform({table["geom_col"]}, 3857), 50))),
        st_astext(st_multi(st_transform({table["geom_col"]}, 3857)))
    from
        {table["schema"]}.{table["table"]}
    where
        {table["key"]} <= {args.end_key}
    and
        {table["key"]} >= {args.start_key}
    and
        description = 'field'
    and
        st_area({table["geom_col"]}::geography) > 1000
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        poly_fetch_all = curs.fetchall()

    for farm in poly_fetch_all:
        row = [farm[0]]
        for month in annotate_months:
            raster_path = f'data_bid/global_monthly_{month[0]}_{months_data[month[1]]}_mosaic/1453-1133_quad.tif'
            multipolygon = loads(farm[1])
            output_path = 'temp_classify.tif'
            clip_raster_with_multipolygon(raster_path, multipolygon, output_path)
            tiff_file = output_path
            raw_poly = farm[2]
            polygon = [(float(item.split(' ')[0]), float(item.split(' ')[1])) for item in raw_poly.strip().split('(')[3].split(')')[0].split(',')]
            highlight_farm(tiff_file, polygon)

            while True:
                plt.ion()
                img = plt.imread('temp_classify.tif')
                plt.title(f'{table["key"]}: {farm[0]}, {months_names[month[1]]}-{month[0]}')
                plt.imshow(img)
                plt.draw()
                plt.pause(1)
                plt.close()
                answer = input()
                if answer == 'y':
                    row.append(1)
                    break
                if answer == 'n':
                    row.append(0)
                    break
                if answer == 'r':
                    continue
        df.loc[len(df)] = row
    pgconn.commit()

    df.to_csv('annotations.csv', index=False, mode='a', header=False)