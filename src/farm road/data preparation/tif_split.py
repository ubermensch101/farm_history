from osgeo import gdal

# Input GeoTIFF file
input_file = '4096_og_data/img1.tif'

# Output directory for tiles
output_dir = '512_tiles/'

# Open the input file
dataset = gdal.Open(input_file)

# Get the width and height of the input file
width = dataset.RasterXSize
height = dataset.RasterYSize

# Define tile size
tile_size_x = 512
tile_size_y = 512

# Loop through the input file and create tiles
for i in range(0, width, tile_size_x):
    for j in range(0, height, tile_size_y):
        # Compute the tile coordinates
        x = i
        y = j
        w = min(tile_size_x, width - i)
        h = min(tile_size_y, height - j)

        # Create the output tile name
        tile_name = f'{output_dir}tile_{x}_{y}_img1.tif'

        # Create a new dataset for the tile
        driver = gdal.GetDriverByName('GTiff')
        tile_dataset = driver.Create(tile_name, w, h, dataset.RasterCount, dataset.GetRasterBand(1).DataType)

        # Set the geotransform and projection
        tile_dataset.SetGeoTransform((x, dataset.GetGeoTransform()[1], 0, y, 0, dataset.GetGeoTransform()[5]))
        tile_dataset.SetProjection(dataset.GetProjection())

        # Read the data from the input file and write it to the tile
        for band_num in range(1, dataset.RasterCount + 1):
            data = dataset.GetRasterBand(band_num).ReadAsArray(x, y, w, h)
            tile_dataset.GetRasterBand(band_num).WriteArray(data)

        # Close the tile dataset
        tile_dataset = None

# Close the input dataset
dataset = None
