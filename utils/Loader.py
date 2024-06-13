import os
import subprocess
import psycopg2

# Path to the folder containing KML files
kml_folder = "Villages"

# PostgreSQL connection parameters
db_name = "telangana_villages"
db_user = "postgres"
db_password = "postgres"

# Function to import KML file into PostgreSQL
def import_kml_to_postgresql(kml_file):
    # Extract table name from the file name
    table_name = str(os.path.splitext(os.path.basename(kml_file))[0])

    # Construct ogr2ogr command
    ogr2ogr_command = [
        "ogr2ogr", "-f", "PostgreSQL",
        f"PG:dbname={db_name} user={db_user} password={db_password}",
        kml_file, "-nln", table_name
    ]

    # Execute ogr2ogr command
    subprocess.run(ogr2ogr_command)

    # Connect to PostgreSQL
    conn = psycopg2.connect(dbname=db_name, user=db_user, password=db_password)
    conn.autocommit = True
    curs = conn.cursor()

    # Drop ogc_fid column and constraints
    try:
        drop_constraint_sql = f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {table_name}_pkey;"
        drop_column_sql = f"ALTER TABLE {table_name} DROP COLUMN IF EXISTS ogc_fid;"
        curs.execute(drop_constraint_sql)
        curs.execute(drop_column_sql)

        # Add new primary key column
        add_column_sql = f"ALTER TABLE {table_name} ADD COLUMN id SERIAL PRIMARY KEY;"
        curs.execute(add_column_sql)

    except Exception as e:
        print(f"Error modifying table {table_name}: {e}")
    finally:
        curs.close()
        conn.close()

count = 0

# Iterate over KML files in the folder
for filename in os.listdir(kml_folder):
    if filename.endswith(".kml"):
        kml_file_path = os.path.join(kml_folder, filename)
        import_kml_to_postgresql(kml_file_path)
        print(count)
        count += 1
        break