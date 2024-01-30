import argparse
import csv
import psycopg2
import matplotlib.pyplot as plt
import rasterio
from shapely.wkt import loads
from rasterio.mask import mask
import pandas as pd

class PGConn:
    def __init__(self, host, port, dbname, user=None, passwd=None):
        self.host = host
        self.port = port
        self.dbname = dbname
        if user is not None:
            self.user = user
        else:
            self.user = ""
        if passwd is not None:
            self.passwd = passwd
        else:
            self.passwd = ""
        self.conn = None

    def connection(self):
        """Return connection to PostgreSQL.  It does not need to be closed
        explicitly.  See the destructor definition below.

        """
        if self.conn is None:
            conn = psycopg2.connect(dbname=self.dbname,
                                    host=self.host,
                                    port=str(self.port),
                                    user=self.user,
                                    password=self.passwd)
            self.conn = conn
            
        return self.conn

pgconn_obj = PGConn(
    "localhost",
    5432,
    "dolr",
    "sameer",
    "swimgood"
)
    
pgconn=pgconn_obj.connection()

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--start_gid", type=int, required=True)
parser.add_argument("-e", "--end_gid", type=int, required=True)

args = parser.parse_args()

sql_query = f"""
select
    gid,
    st_astext(st_transform(st_buffer(geom, 100), 3857)) as geom_text
from
    pilot.dagdagad_farmplots_dedup
where
    gid <= {args.end_gid}
and
    gid >= {args.start_gid}
"""

with pgconn.cursor() as curs:
    curs.execute(sql_query)
    poly_fetch_all = curs.fetchall()

def clip_raster_with_multipolygon(raster_path, multipolygon, output_path):
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Clip the raster with the multipolygon
        out_image, out_transform = mask(src, [multipolygon], crop=True)
        
        # Copy the metadata
        out_meta = src.meta.copy()

        # Update the metadata to match the clipped raster
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        # Write the clipped raster to a new file
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

df = pd.DataFrame(columns=['gid', 'month', 'year', 'crop_presence'])

for farm in poly_fetch_all:
    years = [2022,2023]
    months = ['01','02','03','04','05','06',
              '07','08','09','10','11','12']
    months_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    for year in years:
        for i,month in enumerate(months):
            raster_path = f'data/global_monthly_{year}_{month}_mosaic/1465-1146_quad.tif'
            multipolygon = loads(farm[1])
            output_path = 'temp_classify.tif'
            clip_raster_with_multipolygon(raster_path, multipolygon, output_path)
            while True:
                plt.ion()
                img = plt.imread('temp_classify.tif')
                plt.title(f'gid: {farm[0]}, month: {months_names[i]}, {year}')
                plt.imshow(img)
                plt.draw()
                plt.pause(1)
                plt.close()
                answer = input()
                if answer == 'y':
                    df.loc[len(df)] = [farm[0], months_names[i], year, 1]
                    break
                if answer == 'n':
                    df.loc[len(df)] = [farm[0], months_names[i], year, 0]
                    break
                if answer == 'r':
                    continue

print("Writing results")
df.to_csv('classify_output.csv', index=False)