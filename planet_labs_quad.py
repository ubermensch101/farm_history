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

pgconn_obj.__del__()

def place_monthly_order(mosaic_name):
    order_params = {
        "name": "Basemap order with geometry",
        "source_type": "basemaps",
        "products": [
            {
                "mosaic_name": mosaic_name,
                "quad_ids": [
                    "1465-1146"
                ]
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
        path = pathlib.Path(os.path.join(table["data_dir"], name))
        
        if overwrite or not path.exists():
            print('downloading {} to {}'.format(name, path))
            r = requests.get(url, allow_redirects=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            open(path, 'wb').write(r.content)
            hash = name.strip().split('/')[0]
            os.system(f"cp -r {os.path.join(table['data_dir'], hash)}/global_monthly_* {table['data_dir']}")
            os.system(f"rm -rf {os.path.join(table['data_dir'], hash)}")
        else:
            print('{} already exists, skipping {}'.format(path, name))

tile_months = [
    '2022_05',
    '2022_06',
    '2022_07',
    '2022_08',
    '2022_09',
    '2022_10',
    '2022_11',
    '2022_12',
    '2023_01',
    '2023_02',
    '2023_03',
    '2023_04'
]

order_urls = []
for month in tile_months:
    mosaic_name = f"global_monthly_{month}_mosaic"
    order_url = place_monthly_order(mosaic_name)
    order_urls.append(order_url)

for order_url in order_urls:
    poll_for_success(order_url)
    r = requests.get(order_url, auth=AUTH)
    response = r.json()
    results = response['_links']['results']
    print([r['name'] for r in results])
    download_results(results)
