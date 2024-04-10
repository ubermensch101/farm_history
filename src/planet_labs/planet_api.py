import asyncio
from datetime import datetime
import planet
import os
import time
import json

from config.config import Config
from utils.postgres_utils import PGConn

config = Config()
pgconn_obj = PGConn(config)
pgconn=pgconn_obj.connection()

table = config.setup_details["tables"]["villages"][0]

sql_query = f"""
select
    st_asgeojson(st_transform({table["geom_col"]}, 4674))
from
    {table["schema"]}.{table["table"]}
limit
    1
;
"""
with pgconn.cursor() as curs:
    curs.execute(sql_query)
    polygon = json.loads(curs.fetchall()[0][0])

async def download_and_validate():
    async with planet.Session() as sess:
        cl = sess.client('data')

        sfilter = planet.data_filter.and_filter([
            planet.data_filter.permission_filter(),
            planet.data_filter.date_range_filter(
                'acquired',
                gt=datetime(2022, 6, 1),
                lt=datetime(2022, 6, 8)
            ),
            planet.data_filter.geometry_filter(polygon)
        ])
        
        items = [i async for i in cl.search(['PSScene'], sfilter)][0:2]

        for item_id in [item['id'] for item in items]:
            # get asset description
            item_type_id = 'PSScene'
            asset_type_id = 'basic_analytic_4b'
            asset = await cl.get_asset(item_type_id, item_id, asset_type_id)

            # activate asset
            await cl.activate_asset(asset)

            # wait for asset to become active
            asset = await cl.wait_asset(asset, callback=print)

            # download asset
            path = await cl.download_asset(asset, directory=f"{os.path.dirname(os.path.realpath(__file__))}/../../data/weekly/")

            # validate download file
            cl.validate_checksum(asset, path)

asyncio.run(download_and_validate())
