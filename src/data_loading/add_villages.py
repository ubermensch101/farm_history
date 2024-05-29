import psycopg2
from psycopg2 import sql
import json

# Load GeoJSON data from file
with open('/home/rahul/Downloads/villages-20240520T094644Z-001/villages/CHEVELLA.json') as f:
    data = json.load(f)

# Connect to your PostgreSQL database
conn = psycopg2.connect(database="telangana_villages", user="postgres", password="postgres", host="localhost", port="5432")
cursor = conn.cursor()

# Create a table if not exists
create_table_query = '''
    CREATE TABLE IF NOT EXISTS CHEVELLA (
        id SERIAL PRIMARY KEY,
        parcel_num TEXT,
        remarks TEXT,
        v_name TEXT,
        m_name TEXT,
        d_name TEXT,
        dmv_code TEXT,
        landuse_fr TEXT,
        l_ucode_tiff TEXT,
        st_code INTEGER,
        shape_le_1 NUMERIC,
        shape_area NUMERIC,
        geometry GEOMETRY(POLYGON, 4326)
    )
'''
cursor.execute(create_table_query)
conn.commit()

# Insert data into the table
for feature in data['features']:
    geometry = json.dumps(feature['geometry'])
    properties = feature['properties']
    insert_query = sql.SQL('''
        INSERT INTO CHEVELLA (parcel_num, remarks, v_name, m_name, d_name, dmv_code, landuse_fr, l_ucode_tiff, st_code, shape_le_1, shape_area, geometry)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))
    ''')
    data = (
        properties['Parcel_num'],
        properties['Remarks'],
        properties['V_Name'],
        properties['M_Name'],
        properties['D_Name'],
        properties['DMV_Code'],
        properties['LandUse_Fr'],
        properties['LUCodeTiff'],
        properties['STCode'],
        properties['Shape_Le_1'],
        properties['Shape_Area'],
        geometry
    )
    cursor.execute(insert_query, data)
    conn.commit()

# Close the cursor and the connection
cursor.close()
conn.close()
