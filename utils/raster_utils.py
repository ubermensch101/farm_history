import numpy as np
import rasterio
from rasterio.mask import mask
from PIL import Image, ImageDraw

def clip_raster_with_multipolygon(raster_path, multipolygon, output_path):
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Clip the raster with the multipolygon
        out_image, out_transform = mask(src, [multipolygon], crop=True)
        
        # Copy the metadata
        out_meta = src.meta.copy()

        # Update the metadata to match the clipped raster
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        # Write the clipped raster to a new file
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

def calculate_average_color(tif_path):
    with rasterio.open(tif_path) as src:
        data = src.read()

        if data.shape[0] >= 4:
            red_avg = np.mean(data[0])
            green_avg = np.mean(data[1])
            blue_avg = np.mean(data[2])
            nir_avg = np.mean(data[3])
            return (red_avg, green_avg, blue_avg, nir_avg)
        else:
            raise ValueError("Need 4 bands corresponding to rgb and near-IR")

def highlight_farm(raster_path, polygon):
    with rasterio.open('temp_classify.tif') as dataset:
        pixel_poly = [dataset.index(*coord) for coord in polygon]
    pixel_poly = [(item[1], item[0]) for item in pixel_poly]
    img = Image.open(raster_path)
    draw = ImageDraw.Draw(img)
    draw.polygon(pixel_poly, outline='red')
    img.save('temp_classify.tif')