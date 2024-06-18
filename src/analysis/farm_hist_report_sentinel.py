import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
import geopandas as gpd
from shapely import wkb
from config.config import Config
from utils.postgres_utils import PGConn
from utils.raster_utils import *

## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

months_data = ['01','02','03','04','05','06',
               '07','08','09','10','11','12']
months_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

if __name__ == '__main__':
    print("Generating Report...\n")
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn = pgconn_obj.connection()

    table_config = config.setup_details["tables"]["villages"][2]
    village_name = table_config['table']
    cycle_months = config.setup_details["months"]["agricultural_months"]
    year = config.setup_details["months"]["agricultural_months"][0][0]
    OUTPUT_PATH = os.path.join(CURRENT_DIR, f"farm_hist_report_{village_name}_sentinel.pdf")

    # Create a list of column names for crop presence probabilities
    ## month 1 of agricultural year corresponds to month 5 of the calendar year
    columns = [f"crop_presence_{year}_monthly_{i+1}" for i in range(12)]
    columns_str = ", " .join(columns)

    # Add cropping pattern column
    crop_pattern_column = f"crop_cycle_monthly_{year}_{year+1}"

    # Update the SQL query to fetch geometry, crop presence probabilities, and cropping pattern
    sql_query = f"""
    SELECT {table_config['geom_col']}, {columns_str}, {crop_pattern_column}
    FROM {table_config['schema']}.{table_config['table']}
    WHERE {table_config['filter']}
    """
    
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        records = curs.fetchall()
    
    geometries = [wkb.loads(record[0], hex=True) for record in records]
    crop_probabilities = [record[1:13] for record in records]  # Exclude the geometry and cropping pattern columns
    crop_patterns = [record[13] for record in records]  # Cropping pattern column

    # Create GeoDataFrame and set initial CRS
    gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    crop_prob_df = pd.DataFrame(crop_probabilities, columns=columns)
    crop_pattern_df = pd.Series(crop_patterns, name='crop_pattern')
    gdf = pd.concat([gdf, crop_prob_df, crop_pattern_df], axis=1)

    # Reproject to a metric CRS (e.g., Web Mercator)
    gdf = gdf.to_crs(epsg=3857)

    # Generate the report
    with PdfPages(OUTPUT_PATH) as pdf:
        # Create title page
        title_fig, title_ax = plt.subplots(figsize=(9, 11.69))  # A4 dimensions
        title_ax.axis('off')
        title_ax.text(0.5, 0.5, f"Farm History Report : {table_config['table']}", fontsize=24, ha='center', va='center')
        pdf.savefig(title_fig)
        plt.close(title_fig)

        # Loop through columns and create plots
        for i, month in enumerate(cycle_months):
            fig = plt.figure(figsize=(9, 11.69))  # A4 dimensions
            ax_plot = fig.add_axes([0.1, 0.45, 0.7, 0.5])  # Adjusted plot area to make room for legend
            ax_table = fig.add_axes([0.1, 0.1, 0.8, 0.25])  # Smaller table area
            
            # Plot
            gdf.plot(column=columns[i], ax=ax_plot, cmap='YlGn', edgecolor='black')
            crop_month = months_names[month[1]]
            ax_plot.set_title(f'Farm Plots - {table_config["table"]} - ({crop_month.upper()})')
            ax_plot.set_xlabel('Longitude')
            ax_plot.set_ylabel('Latitude')
            
            # Manual creation of legend with specified colors and labels
            bins = [0, 0.25, 0.5, 0.75, 1]
            colors = plt.cm.YlGn(np.linspace(0, 1, len(bins)))
            legend_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
            handles = [plt.Line2D([], [], color=colors[i], marker='o', linestyle='', markersize=10) for i in range(len(legend_labels))]
            
            # Adjust the position of the legend to be inside the plot area without overlapping
            ax_plot.legend(handles, legend_labels, title='Crop Presence Probability', bbox_to_anchor=(1.04, 0.5), loc='center left')

            # Crop presence statistics
            num_farms_with_crop = len(gdf[gdf[columns[i]] > 0.5])
            num_farms_without_crop = len(gdf[gdf[columns[i]] <= 0.5])
            area_with_crop = gdf[gdf[columns[i]] > 0.2].area.sum() / 10000  # Convert to hectares
            area_without_crop = gdf[gdf[columns[i]] <= 0.2].area.sum() / 10000  # Convert to hectares
            
            total_area = round(gdf.area.sum() / 10000, 6)  # Convert to hectares
            percent_area_with_crop = round((area_with_crop / total_area) * 100, 2)
            area_with_crop = round(area_with_crop, 2)
            area_without_crop = round(area_without_crop, 2)  
              # Clip to 5 decimal places

            # Create a table subplot
            table_data = {"# Farms with crop": num_farms_with_crop,
                          "# Farms without crop": num_farms_without_crop,
                          "Area with crop (ha)": area_with_crop,
                          "Area without crop (ha)": area_without_crop,
                          "% Area with crop": percent_area_with_crop}
            ax_table.axis('off')  # Turn off axis for the table subplot

            # Transpose the table data
            transposed_data = list(zip(table_data.keys(), table_data.values()))
            
            # Create the table with increased font size and bold headers
            data_table = ax_table.table(cellText=[list(item) for item in transposed_data], 
                                        colLabels=['Metric', 'Value'], 
                                        loc='center', 
                                        cellLoc='center')
            data_table.auto_set_font_size(False)
            data_table.set_fontsize(12)  # Increase font size
            data_table.scale(1, 1.5)  # Adjust the scale to fit the text better
            
            # Bold the column headers
            for (i, j), cell in data_table.get_celld().items():
                if i == 0:  # Header cells
                    cell.set_text_props(fontweight='bold')

            # Save the current figure into the PDF
            pdf.savefig(fig)
            plt.close(fig)
        
        # Cropping pattern statistics and plot
        crop_pattern_counts = gdf['crop_pattern'].value_counts().sort_index()
        unique_patterns = crop_pattern_counts.index.tolist()
        # unique_patterns.sort()
        
        # Create a color map for the cropping patterns
        pattern_cmap = sns.color_palette("hsv", len(unique_patterns))
        pattern_color_dict = dict(zip(unique_patterns, pattern_cmap))
        
        # Assign colors to the cropping patterns in the GeoDataFrame
        gdf['pattern_color'] = gdf['crop_pattern'].map(pattern_color_dict)
        
        # Create a new figure for the cropping pattern plot
        pattern_fig = plt.figure(figsize=(9, 11.69))  # A4 dimensions
        pattern_ax_plot = pattern_fig.add_axes([0.1, 0.45, 0.7, 0.5])  # Adjusted plot area to make room for legend
        pattern_ax_table = pattern_fig.add_axes([0.1, 0.1, 0.8, 0.25])  # Smaller table area
        
        # Plot the farm plots with cropping pattern colors
        gdf.plot(ax=pattern_ax_plot, color=gdf['pattern_color'], edgecolor='black')
        pattern_ax_plot.set_title('Farm Plots - Cropping Patterns')
        pattern_ax_plot.set_xlabel('Longitude')
        pattern_ax_plot.set_ylabel('Latitude')
        
        # Create legend for cropping patterns
        handles = [plt.Line2D([], [], color=color, marker='o', linestyle='', markersize=10) 
                   for color in pattern_color_dict.values()]
        labels = [pattern for pattern in pattern_color_dict.keys()]
        pattern_ax_plot.legend(handles, labels, title='Cropping Pattern', bbox_to_anchor=(1.04, 0.5), loc='center left')
        
        # Create table with cropping pattern statistics
        table_data = {
            'Pattern': crop_pattern_counts.index,
            'Count': crop_pattern_counts.values
        }
        table_df = pd.DataFrame(table_data)
        
        pattern_ax_table.axis('off')  # Turn off axis for the table subplot

        # Transpose the table data
        transposed_data = list(zip(table_df['Pattern'], table_df['Count']))
        
        # Create the table with increased font size and bold headers
        data_table = pattern_ax_table.table(cellText=[list(item) for item in transposed_data], 
                                            colLabels=['Cropping Pattern', 'Count'], 
                                            loc='center', 
                                            cellLoc='center')
        data_table.auto_set_font_size(False)
        data_table.set_fontsize(12)  # Increase font size
        data_table.scale(1, 1.5)  # Adjust the scale to fit the text better
        
        # Bold the column headers
        for (i, j), cell in data_table.get_celld().items():
            if i == 0:  # Header cells
                cell.set_text_props(fontweight='bold')
        
        # Save the cropping pattern plot figure into the PDF
        pdf.savefig(pattern_fig)
        plt.close(pattern_fig)

    print(f"Report generated at {OUTPUT_PATH}")
