import os
import rasterio
from rasterio.merge import merge
from rasterio.transform import Affine

# List of paths to the GeoTIFF files
tif=os.listdir('Split_mask256_new_sep_old')
tif_files_temp=[]
for f in tif:
    filename=os.path.join('Split_mask256_new_sep_old', f)
    tif_files_temp.append(filename)
tif_files=[f for f in tif_files_temp if f.endswith('tif')]
# List to store opened raster files
src_files_to_mosaic = []

# Open each GeoTIFF file and extract its geotransform
for tif_file in tif_files:
    src = rasterio.open(tif_file)
    src_files_to_mosaic.append(src)

# Merge the raster files into a single raster
mosaic, out_trans = merge(src_files_to_mosaic)

# Combine the geotransforms to create the geotransform for the merged raster
# Here, we simply use the geotransform of the first raster file
# You may need to adjust this depending on your specific requirements
out_trans = src_files_to_mosaic[0].transform

# Update the metadata of the merged raster with the combined geotransform
out_meta = src_files_to_mosaic[0].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans
})

# Save the merged raster to a new GeoTIFF file
output_path = 'merged256_new_sep_old.tif'
with rasterio.open(output_path, "w", **out_meta) as dest:
    dest.write(mosaic)

print("Merged GeoTIFF file saved successfully.")