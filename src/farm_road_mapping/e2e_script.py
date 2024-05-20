import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from model import DeepLabv3
from torch.utils.data import DataLoader
from osgeo import gdal
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.plot import show
import json


## Input Image: Georeferenced 2048 x 2048 tif file
input_path='sep_old.tif'

## Output Path of final Georeferenced Mask Image 
output_path='mask.tif'

## Meta Directory for saving split Images
os.makedirs('Splits256_new_sep_old', exist_ok=True)
meta_dir='Splits256_new_sep_old'

meta_json='geotransform.json'

## Meta Directory for saving split masks
os.makedirs('Split_mask256_new_sep_old', exist_ok=True)
meta_mask='Split_mask256_new_sep_old'

## Trained Model 
model_dir='NEW_MODEL256.pt'

INPUT_SIZE=256
model=DeepLabv3()
checkpoint=torch.load(model_dir)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
device=('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

transforms=transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), 2),
        transforms.ToTensor()
    ])

input_ds = gdal.Open(input_path)
geotransform = input_ds.GetGeoTransform()
block_width = input_ds.RasterXSize
block_height = input_ds.RasterYSize

for block_x in range(0, block_width, INPUT_SIZE):
    for block_y in range(0, block_height, INPUT_SIZE):

        output_path = os.path.join(meta_dir, f'block_{block_x}_{block_y}.tif')
        output_ds = gdal.GetDriverByName('GTiff').Create(output_path, INPUT_SIZE, INPUT_SIZE, input_ds.RasterCount, input_ds.GetRasterBand(1).DataType)
        output_ds.SetGeoTransform((
            geotransform[0] + block_x * geotransform[1],
            geotransform[1],
            0.0,
            geotransform[3] + block_y * geotransform[5],
            0.0,
            geotransform[5]
        ))
        for band in range(1, input_ds.RasterCount + 1):
            input_band = input_ds.GetRasterBand(band)
            output_band = output_ds.GetRasterBand(band)
            output_band.ReadAsArray(block_x, block_y, INPUT_SIZE, INPUT_SIZE)
            output_band.WriteArray(input_band.ReadAsArray(block_x, block_y, INPUT_SIZE, INPUT_SIZE))
        output_ds = None
        input_band = None
        output_band = None
input_ds = None


split_files=os.listdir(meta_dir)

count=0
for f in split_files:
    count+=1
    image_path=os.path.join(meta_dir, f)
    img=Image.open(image_path).convert('RGB')
    img=transforms(img)
    img=torch.tensor(img).unsqueeze(0)
    img=img.to(device)
    with torch.no_grad():
        output=model(img)
        output=torch.argmax(output, dim=1)

    output = output.squeeze(0).cpu().numpy()
    output=output*255
    output=np.uint8(output)
    kernel = np.ones((5,5),np.uint8)
    dilated_image = 255-cv2.dilate(255-output, kernel, iterations=1)
    out=Image.fromarray((dilated_image).astype(np.uint8))
    save_path=os.path.join(meta_mask, f'output_{count}.tif')
    out.save(save_path)
    ## Input Image Path: image_path
    ## Mask Image Path: save_path

    #Create JSON for GeoTransform Data for the src image

    # dataset = gdal.Open(image_path)
    # geotransform = dataset.GetGeoTransform()
    # srs = dataset.GetProjection()
    # georef_data = {
    # "geotransform": geotransform,
    # "srs": srs}
    # json_data = json.dumps(georef_data, indent=4)
    # with open(meta_json, 'w') as json_file:
    #     json_file.write(json_data)
    # dataset=None

    # # Copy the GeoTransform to dst raster

    # with open(meta_json, 'r') as f:
    #     geojson_data = json.load(f)

    # geotransform = geojson_data['geotransform']
    # srs = geojson_data['srs']

    # with rasterio.open(save_path, 'r+') as src:
    #     src.transform = geotransform
    #     # src.crs = srs
    #     src.close()


    with rasterio.open(image_path) as src:
        transform = src.transform
        crs = src.crs
        dtype = src.dtypes[0] 
        with rasterio.open(save_path, 'r+') as dst:
            dst.transform = transform
    

    # src_ds = gdal.Open(image_path)
    # dst_ds = gdal.Open(save_path)
    # src_geotransform = src_ds.GetGeoTransform()
    # src_projection = src_ds.GetProjection()
    # dst_ds.SetGeoTransform(src_geotransform)
    # dst_ds.SetProjection(src_projection)
    # dst_ds.FlushCache()
    # dst_ds = None



# split_mask_files = os.listdir(meta_mask)
# driver = gdal.GetDriverByName('GTiff')
# output_tif = driver.Create(output_path, 2048, 2048, 1, gdal.GDT_Byte)
# output_tif.SetGeoTransform(geotransform)
# for split_mask_file in [f for f in split_mask_files if f.endswith('tif')]:
#     split_mask_path = os.path.join(meta_mask, split_mask_file)
#     split_mask = gdal.Open(split_mask_path)
#     split_mask_data = split_mask.ReadAsArray()
#     output_tif.GetRasterBand(1).WriteArray(split_mask_data)


for f in os.listdir(meta_mask):
    if not f.endswith('.tif'):
        im=os.path.join(meta_mask,f)
        os.remove(im)

# # # split_mask.FlushCache()
# # # output_tif.FlushCache()
# # # output_tif = None



# # Get all filenames ending with .tif
# tif_files = [f for f in os.listdir(meta_mask) if f.endswith(".tif")]

# # Check if there are any TIF files
# if not tif_files:
#     print("No TIF files found in", meta_mask)
#     exit()

# # Define the output filename
# output_file = "merged.tif"

# # Open the first TIF file to get reference information
# with open(os.path.join(meta_mask, tif_files[0])) as first_file:
#     # Get profile of the first file (assumes all files have same properties)
#     profile = first_file.profile

# # Update profile with desired options (optional)
# profile.update(
#     compress='lzw',  # Lossless compression (adjust as needed)
#     count=len(tif_files),  # Number of bands in the output
# )

# # List to store opened datasets
# datasets = []

# # Open all TIF files
# for filename in tif_files:
#     filepath = os.path.join(meta_mask, filename)
#     datasets.append(open(filepath))

# # Perform the merge operation
# merged_data, merged_transform = merge(datasets, bounds_first=True)

# # Write the merged data and transform to the output file
# with open(output_file, 'w', **profile) as dst:
#     dst.write(merged_data)
#     dst.transform = merged_transform

# print(f"Merged TIF files saved to: {output_file}")

# # Close all datasets
# for dataset in datasets:
#     dataset.close()


src_files_to_mosaic = []
for fp in os.listdir(meta_mask):
    fp_path=os.path.join(meta_mask, fp)
    src = rasterio.open(fp_path)
    src_files_to_mosaic.append(src)
mosaic, out_trans = merge(src_files_to_mosaic)
out_meta = src.meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans,
    "crs": src_files_to_mosaic[0].crs  # Assuming all rasters have the same CRS
})
with rasterio.open(output_path, "w", **out_meta) as dest:
    dest.write(mosaic)
print("Mask image created successfully.")





