import copy
import json
import math
import os
import pathlib
import time
import psycopg2
import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry

from config.config import Config
from utils.postgres_utils import PGConn

PL_API_KEY = 'PLAK721cf7576b5f4835bcd0f2dfe9a7a395'
ORDERS_API_URL = 'https://api.planet.com/compute/ops/orders/v2'
BASEMAP_API_URL = 'https://api.planet.com/basemaps/v1/mosaics'

SESSION = requests.Session()

SESSION.auth = (PL_API_KEY, '')
AUTH = HTTPBasicAuth(PL_API_KEY, '')

retries = Retry(total=10, backoff_factor=1, status_forcelist=[429])
SESSION.mount('https://', HTTPAdapter(max_retries=retries))

config = Config()
pgconn_obj = PGConn(config)
pgconn=pgconn_obj.connection()

table = config.setup_details["tables"]["villages"][0]
data_directory = f"{os.path.dirname(os.path.realpath(__file__))}/../../quads/{table['table']}"

sql_query = f"""
select
    st_astext(st_transform({table['geom_col']}, 4674)) as geom_text
from
    {table["schema"]}.{table["table"]}
limit
    1
;
"""

with pgconn.cursor() as curs:
    curs.execute(sql_query)
    poly_fetch = curs.fetchone()[0]

poly_split = poly_fetch.split('(')
poly_coords = poly_split[3]
poly_coords = poly_coords.split(')')[0]
poly_point_coords = poly_coords.split(',')

points = []
for pt_coords in poly_point_coords:
    pt_split = pt_coords.split(' ')
    lat = float(pt_split[0])
    long = float(pt_split[1])
    points.append([lat, long])
print(points)

pgconn_obj.__del__()

def place_monthly_order(mosaic_name, points):
    order_params = {
        "name": "Basemap order with geometry",
        "source_type": "basemaps",
        "products": [
            {
                "mosaic_name": mosaic_name,
                "geometry":{
                "type": "Polygon",
                "coordinates":[
                    points
                ]
                }
            }
        ]
    }
    
    paramRes = requests.post(ORDERS_API_URL,
        data=json.dumps(order_params),
        auth=AUTH,
        headers={'content-type': 'application/json'}
    )

    print(paramRes.text)

    order_id = paramRes.json()['id']
    order_url = ORDERS_API_URL + '/' + order_id
    return order_url

def poll_for_success(order_url, num_loops=30):
    count = 0
    while(count < num_loops):
        count += 1
        r = requests.get(order_url, auth=AUTH)
        response = r.json()
        state = response['state']
        print(state)
        end_states = ['success', 'failed', 'partial']
        if state in end_states:
            break
        time.sleep(10)

def download_results(results, overwrite=False):
    results_urls = [r['location'] for r in results]
    results_names = [r['name'] for r in results]
    print('{} items to download'.format(len(results_urls)))
    
    for url, name in zip(results_urls, results_names):
        path = pathlib.Path(os.path.join(data_directory, name))
        
        if overwrite or not path.exists():
            print('downloading {} to {}'.format(name, path))
            r = requests.get(url, allow_redirects=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            open(path, 'wb').write(r.content)
            hash = name.strip().split('/')[0]
            os.system(f"cp -r {os.path.join(data_directory, hash)}/global_monthly_* {data_directory}")
            os.system(f"rm -rf {os.path.join(data_directory, hash)}")
        else:
            print('{} already exists, skipping {}'.format(path, name))

order_urls = []
for month in range(1, 13):
    if month < 10:
        month = "0" + str(month)
    else:
        month = str(month)
    mosaic_name = f"global_monthly_2022_{month}_mosaic"
    order_url = place_monthly_order(mosaic_name, points)
    order_urls.append(order_url)
    mosaic_name = f"global_monthly_2023_{month}_mosaic"
    order_url = place_monthly_order(mosaic_name, points)
    order_urls.append(order_url)

for order_url in order_urls:
    poll_for_success(order_url)
    r = requests.get(order_url, auth=AUTH)
    response = r.json()
    results = response['_links']['results']
    print([r['name'] for r in results])
    download_results(results)