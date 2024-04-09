import json
from osgeo import gdal

# Path to the GeoTIFF file
file_path = '512_spilts/512_spilts.0.tif'

# Open the GeoTIFF file
dataset = gdal.Open(file_path)

if dataset is None:
    print("Error: Could not open the file.")
    exit(1)

# Get the geotransform parameters
geotransform = dataset.GetGeoTransform()

if geotransform is None:
    print("Error: No geotransform found.")
    exit(1)

# Get the spatial reference system (SRS)
srs = dataset.GetProjection()

if srs is None:
    print("Error: No SRS found.")
    exit(1)

# Create a dictionary to store the georeference data
georef_data = {
    "geotransform": geotransform,
    "srs": srs
}

# Convert the dictionary to JSON
json_data = json.dumps(georef_data, indent=4)

# Save the JSON data to a file
with open('georef_data_1.json', 'w') as json_file:
    json_file.write(json_data)

# Close the dataset
dataset = None
