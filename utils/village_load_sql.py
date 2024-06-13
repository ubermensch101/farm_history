import os
import subprocess
import psycopg2

# Path to the folder containing KML files
kml_folder = "Villages"

# PostgreSQL connection parameters
db_name = "telangana_villages"
db_user = "postgres"
db_password = "postgres"
db_host = "localhost"  # Change if your database is hosted elsewhere
db_port = "5432"       # Default PostgreSQL port

# Function to import KML file into PostgreSQL
def import_kml_to_postgresql(kml_file):
    # Extract table name from the file name
    table_name = str(os.path.splitext(os.path.basename(kml_file))[0])

    # Construct ogr2ogr command
    ogr2ogr_command = [
        "ogr2ogr", "-f", "PostgreSQL",
        f"PG:dbname={db_name} user={db_user} password={db_password} host={db_host} port={db_port}",
        kml_file, "-nln", table_name
    ]

    # Execute ogr2ogr command
    try:
        subprocess.run(ogr2ogr_command, check=True)

        # Modify the table structure in PostgreSQL
        with psycopg2.connect(dbname=db_name, user=db_user, password=db_password, host=db_host, port=db_port) as conn:
            with conn.cursor() as curs:
                # Drop the existing ogc_fid column if it exists
                drop_column_sql = f"ALTER TABLE {table_name} DROP COLUMN IF EXISTS ogc_fid;"
                curs.execute(drop_column_sql)
                
                # Add a new sequential column starting from 1 as the primary key
                add_column_sql = f"""
                ALTER TABLE {table_name}
                ADD COLUMN id SERIAL PRIMARY KEY;
                """
                curs.execute(add_column_sql)

    except subprocess.CalledProcessError as e:
        print(f"Error importing KML file {kml_file}: {e}")
    except psycopg2.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

count = 0
# Iterate over KML files in the folder
for filename in os.listdir(kml_folder):
    if filename.endswith(".kml"):
        kml_file_path = os.path.join(kml_folder, filename)
        import_kml_to_postgresql(kml_file_path)
        print(f"Imported {filename} with new sequential primary key.")
        count += 1
        

print(f"Total files processed: {count}")
