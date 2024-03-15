import numpy as np
import rasterio
from rasterio.mask import mask
from PIL import Image, ImageDraw
import os
from shapely.geometry import shape

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

def get_available_rasters_with_intersection(directory, polygon):
    """
    Get a list of available raster files in the specified directory
    that intersect with the given polygon.

    Parameters:
        directory (str): Path to the directory containing raster files.
        polygon (shapely.geometry.Polygon): Polygon geometry.

    Returns:
        List[str]: List of available raster file names that intersect
                   with the given polygon.
    """
    raster_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tif'):  
                raster_path = os.path.join(root, file)
                with rasterio.open(raster_path) as src:
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
                        raster_files.append(raster_path)
    return raster_files
