import numpy as np
import os
import cv2
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, shape, box
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

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

def compute_hue_features(tif_path):
    """
    # 3 bands corresponding to RGB and the 4th band is essentially
    a bit mask (Since farmplots are not rectangles, some values will
    be 0 i.e. unused)
    """
    with rasterio.open(tif_path) as src:
        data = src.read()
        if data.shape[0] >= 4:
            rgb_bands = np.array(data)
            pixel_mask = rgb_bands[3]
            pixel_mask = pixel_mask.astype(bool)
            rgb_image = np.stack([rgb_bands[0], rgb_bands[1], rgb_bands[2]], axis=-1)
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            hue = hsv_image[:,:,0]
            if len(hue[pixel_mask]) == 0:
                return 0, 0
            hue_mean = np.mean(hue[pixel_mask])
            hue_stddev = np.std(hue[pixel_mask])
            return hue_mean, hue_stddev
        if data.shape[0] == 3:
            # Uses the green band as the pixel mask if no pixel mask available
            rgb_bands = np.array(data)
            pixel_mask = rgb_bands[1]
            pixel_mask = pixel_mask.astype(bool)
            rgb_image = np.stack([rgb_bands[0], rgb_bands[1], rgb_bands[2]], axis=-1)
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            hue = hsv_image[:,:,0]
            if len(hue[pixel_mask]) == 0:
                return 0, 0
            hue_mean = np.mean(hue[pixel_mask])
            hue_stddev = np.std(hue[pixel_mask])
            return hue_mean, hue_stddev
        else:
            raise ValueError("Need 3 bands corresponding to rgb and a pixel mask\nOr at least 3 bands for rgb")

def highlight_farm(raster_path, polygon, output_path=None):
    with rasterio.open(raster_path) as dataset:
        pixel_poly = [dataset.index(*coord) for coord in polygon]
    pixel_poly = [(item[1], item[0]) for item in pixel_poly]
    img = Image.open(raster_path)
    draw = ImageDraw.Draw(img)
    draw.polygon(pixel_poly, outline='red')
    if output_path is not None:
        img.save(output_path)
    return img

def super_clip(directory, year, month, polygon, output_path):
    if year == None:
        fortnight_no = month
        subdir = str(fortnight_no)
    else:
        subdir = f'global_monthly_{year}_{month}_mosaic'
    available_quads = os.listdir(directory)
    for quad in available_quads:
        subdir_path = os.path.join(directory, quad, subdir)
        files = os.listdir(subdir_path)
        for file in files:
            if file.endswith('quad.tif') or file.endswith('response.tiff'):
                with rasterio.open(os.path.join(subdir_path, file)) as src:
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
                        raster_path = os.path.join(subdir_path, file)
                        return clip_raster_with_multipolygon(raster_path, polygon, output_path)
    raise Exception("Couldn't find intersecting raster")
                    


"""
Crop farm image from raster image
"""

def crop_highlighted_farm(raster_path:str, polygon:list):
    # Convert polygon to GeoJSON-like geometry
    polygon_geojson = mapping(shape({"type": "Polygon", "coordinates": [polygon]}))
    
    with rasterio.open(raster_path) as src:
        # Get raster bounds as a shapely.geometry.box object
        raster_bounds = box(*src.bounds)

        # Create shapely geometry object for the provided polygon
        polygon_shape = shape(polygon_geojson)

        # Check if the polygon intersects with the raster bounds
        if raster_bounds.intersects(polygon_shape):
            # Mask the raster with the polygon
            out_image, out_transform = mask(src, [polygon_geojson], crop=True)
            out_meta = src.meta.copy()

            # Update the metadata to match the clipped raster
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})

            # Write the clipped raster to a new file
            output_path = raster_path.replace('.tif', '_cropped.tif')
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
            
            return output_path
        else:
            raise ValueError("Polygon does not intersect with raster bounds.")


"""
Map the polygon cropped farm image to 
a rectangular representation
"""

def remove_padding(raster_path:str, output_path: str):
    image = plt.imread(raster_path)
    n_channels = image.shape[2]
    mask = np.any(image != 0, axis=-1)
    non_padding_pixels = image[mask]
    one_d_vector = non_padding_pixels.reshape(-1, n_channels)
    num_pixels = len(one_d_vector)

    def find_nearest_factors(n):
        for i in range(int(np.sqrt(n)), int(np.sqrt(n))//3, -1):
            if n % i == 0:
                return i, n // i
        return None, None       ## Representation into rectangle isn't possible

    # Find the nearest rectangular shape
    width, height = find_nearest_factors(num_pixels)

    if width==None or height==None:
        avg_value = np.mean(one_d_vector, axis=0)  # Calculate average across each channel
        while width==None or height==None:
            one_d_vector = np.append(one_d_vector, [avg_value], axis=0)
            num_pixels += 1
            width, height = find_nearest_factors(num_pixels)

    # Reshape into the nearest rectangular representation, preserving channel information
    rectangular_representation = one_d_vector.reshape((width, height, n_channels))   
    pil_image = Image.fromarray(rectangular_representation.astype('uint8'))
    pil_image.save(output_path)