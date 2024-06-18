import pandas as pd
import matplotlib.pyplot as plt
from config.config import Config  # Adjust the import as per your project structure
from utils.postgres_utils import PGConn  # Adjust the import as per your project structure

def fetch_data_from_db(table_name):
    config = Config()  # Adjust as per your project structure
    pgconn_obj = PGConn(config)
    pgconn = pgconn_obj.connection()

    # SQL query to fetch data and calculate area (in square meters)
    sql_query = f"""
    SELECT crop_cycle_22_23, kcluster, ST_Area(wkb_geometry::geography) AS area_sq_meters
    FROM {table_name}
    """

    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        records = curs.fetchall()

    # Create a pandas dataframe from the fetched records
    df = pd.DataFrame(records, columns=['crop_cycle_22_23', 'kcluster', 'area_sq_meters'])
    
    # Convert area from square meters to hectares and acres without rounding
    df['area_hectares'] = df['area_sq_meters'] * 0.0001
    df['area_acres'] = df['area_sq_meters'] * 0.000247105
    
    return df

if __name__ == "__main__":
    # Replace 'your_table_name' with your actual table name
    table_name = 'hunsa_cleaned'
    
    # Fetch data from the database
    df = fetch_data_from_db(table_name)
    
    # Group by crop_cycle_22_23 and calculate total area in hectares and acres
    grouped_by_crop_cycle = df.groupby('crop_cycle_22_23').sum().reset_index()
    grouped_by_kcluster = df.groupby('kcluster').sum().reset_index()
    
    # Round the grouped values to integers for display
    grouped_by_crop_cycle['area_hectares'] = grouped_by_crop_cycle['area_hectares'].round(0).astype(int)
    grouped_by_crop_cycle['area_acres'] = grouped_by_crop_cycle['area_acres'].round(0).astype(int)
    grouped_by_kcluster['area_hectares'] = grouped_by_kcluster['area_hectares'].round(0).astype(int)
    grouped_by_kcluster['area_acres'] = grouped_by_kcluster['area_acres'].round(0).astype(int)

    # Print total area in hectares and acres for each crop cycle
    print("Total Area by Crop Cycle (in Hectares and Acres):\n")
    for index, row in grouped_by_crop_cycle.iterrows():
        crop_cycle = row['crop_cycle_22_23']
        area_hectares = row['area_hectares']
        area_acres = row['area_acres']
        print(f"{crop_cycle}:")
        print(f"   - Area in Hectares: {area_hectares:.0f}")  # Display as integer without decimals
        print(f"   - Area in Acres: {area_acres:.0f}")  # Display as integer without decimals
        print()

    # Print total area in hectares and acres for each kcluster
    print("\nTotal Area by KCluster (in Hectares and Acres):\n")
    for index, row in grouped_by_kcluster.iterrows():
        kcluster = row['kcluster']
        area_hectares = row['area_hectares']
        area_acres = row['area_acres']
        print(f"{kcluster}:")
        print(f"   - Area in Hectares: {area_hectares:.0f}")  # Display as integer without decimals
        print(f"   - Area in Acres: {area_acres:.0f}")  # Display as integer without decimals
        print()

    # Format the values to be displayed in the tables
    formatted_crop_cycle = grouped_by_crop_cycle[['crop_cycle_22_23', 'area_hectares', 'area_acres']].astype(str).values
    formatted_kcluster = grouped_by_kcluster[['kcluster', 'area_hectares', 'area_acres']].astype(str).values

    # Plotting the results in table format using Matplotlib
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Table for Crop Cycle
    table_crop_cycle = axes[0].table(cellText=formatted_crop_cycle,
                                     colLabels=['Crop Cycle', 'Area (Hectares)', 'Area (Acres)'],
                                     loc='upper center',
                                     cellLoc='center',
                                     colColours=['#f2f2f2']*3)  # Light gray column headers

    table_crop_cycle.auto_set_font_size(False)
    table_crop_cycle.set_fontsize(12)
    table_crop_cycle.scale(1.2, 1.2)

    axes[0].axis('off')
    axes[0].set_title('Total Area Predicted By LSTM Model', fontsize=14, fontweight='bold', color='navy')

    # Table for KCluster
    table_kcluster = axes[1].table(cellText=formatted_kcluster,
                                   colLabels=['KCluster', 'Area (Hectares)', 'Area (Acres)'],
                                   loc='upper center',
                                   cellLoc='center',
                                   colColours=['#f2f2f2']*3)  # Light gray column headers

    table_kcluster.auto_set_font_size(False)
    table_kcluster.set_fontsize(12)
    table_kcluster.scale(1.2, 1.2)

    axes[1].axis('off')
    axes[1].set_title('Total Area by KCluster', fontsize=14, fontweight='bold', color='navy')

    plt.tight_layout()
    plt.show()
