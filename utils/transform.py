import psycopg2

# Connect to the PostgreSQL database
def connect_to_database():
    try:
        conn = psycopg2.connect(
            dbname="telangana",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432"
        )
        print("Connected to the database")
        return conn
    except Exception as e:
        print("Error connecting to the database:", e)
        return None

# Function to transform geometries in all tables
def transform_geometries(conn):
    try:
        cur = conn.cursor()
        
        # Get a list of all tables in the public schema
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = current_schema();
        """)
        tables = cur.fetchall()
        tables=tables[3:]
        Count=0
        # Loop through each table and transform geometries
        for table in tables:
            table_name = table[0]
            print(table_name)
            cur.execute(f"""
                ALTER TABLE {table_name}
                DROP COLUMN wkb_geometry;
                ALTER TABLE {table_name}
                RENAME COLUMN wkb_geom TO wkb_geometry
               
            """)
            print(Count)
            Count=Count+1
  #          print(f"Geometries transformed for table: {table_name}")

        conn.commit()
        print("Transformation completed successfully")
    except Exception as e:
        conn.rollback()
        print("Error transforming geometries:", e)

# Main function
def main():
    conn = connect_to_database()
    if conn:
        transform_geometries(conn)
        conn.close()

if __name__ == "__main__":
    main()
