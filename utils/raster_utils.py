import numpy as np
import os
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape
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

        if data.shape[0] == 4:
            band_averages = []
            for i in range(4):
                band_averages.append(np.mean(data[i]))
            return band_averages # red, green, blue, nir
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

def super_clip(directory, year, month, polygon, output_path):
    available_quads = os.listdir('quads')
    for quad in available_quads:
        month_path = os.path.join('quads', quad, f'global_monthly_{year}_{month}_mosaic')
        files = os.listdir(month_path)
        for file in files:
            if file.endswith('quad.tif'):
                with rasterio.open(os.path.join(month_path, file)) as src:
                    raster_bbox = src.bounds
                    raster_polygon = shape({
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [raster_bbox.left, raster_bbox.bottom],
                                [raster_bbox.right, raster_bbox.bottom],
                                [raster_bbox.right, raster_bbox.top],
                                [raster_bbox.left, raster_bbox.top],
                                [raster_bbox.left, raster_bbox.bottom]
                            ]
                        ]
                    })
                    if raster_polygon.intersects(polygon):
                        raster_path = os.path.join(month_path, file)
                        return clip_raster_with_multipolygon(raster_path, polygon, output_path)