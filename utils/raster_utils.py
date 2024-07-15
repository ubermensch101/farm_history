import numpy as np
from PIL import ImageEnhance, Image
import os
import cv2
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, shape, box
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def save_multidimension_raster(data, output_path):
    """
    Description:    This function saves a multidimensional raster to a file.

    Args:
                data: np.ndarray     - The multidimensional raster data, shape should be (bands, heigth, width)
                output_path: str     - The path to save the raster to
    """
    metadata = {
        'driver': 'GTiff',
        'dtype': data.dtype,
        'count': data.shape[0],
        'width': data.shape[2],
        'height': data.shape[1],
        'crs': 'EPSG:4326',

    }
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(data)


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
    a bit mask
    (Since farmplots are not rectangles, some values will be 0 i.e. unused)
    """
    with rasterio.open(tif_path) as src:
        data = src.read()
        if data.shape[0] >= 4:
            rgb_bands = np.array(data)
            pixel_mask = rgb_bands[3]
            pixel_mask = ~pixel_mask.astype(bool)
            rgb_image = np.stack([rgb_bands[0], rgb_bands[1], rgb_bands[2]], axis=-1)
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            hue = hsv_image[:,:,0]
            if len(hue[pixel_mask]) == 0:
                return -1,-1
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
            if np.all(hue[pixel_mask]) == 0:
                return 0, 0
            hue_mean = np.mean(hue[pixel_mask])
            hue_stddev = np.std(hue[pixel_mask])
            return hue_mean, hue_stddev
        else:
            raise ValueError("Need 3 bands corresponding to rgb and a pixel mask\nOr at least 3 bands for rgb")


def compute_hue_ir_features(tif_path):
    """
    # 4 bands corresponding to RGB + IR and the 5th band (if there) is essentially
    a cloud bit mask
    (Since farmplots are not rectangles, some values will be 0 i.e. unused)
    """
    with rasterio.open(tif_path) as src:
        data = src.read()
        if data.shape[0] >= 5:
            bands = np.array(data)
            ir_bands = bands[3]
            pixel_mask = bands[4]
            pixel_mask = ~pixel_mask.astype(bool)
            rgb_image = np.stack([bands[0], bands[1], bands[2]], axis=-1)
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            hue = hsv_image[:,:,0]
            non_zero_pixels = np.all(hue[pixel_mask]==0)
            if non_zero_pixels:
                return -1,-1, -1, -1
            hue_mean = np.mean(hue[pixel_mask])
            hue_stddev = np.std(hue[pixel_mask])
            ir_mean = np.mean(ir_bands[pixel_mask])
            ir_stddev = np.std(ir_bands[pixel_mask])
            return hue_mean,hue_stddev,ir_mean, ir_stddev
        
        if data.shape[0] == 3:
            # Uses the green band as the pixel mask if no pixel mask available
            rgb_bands = np.array(data)
            pixel_mask = rgb_bands[1]
            pixel_mask = pixel_mask.astype(bool)
            rgb_image = np.stack([rgb_bands[0], rgb_bands[1], rgb_bands[2]], axis=-1)
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            hue = hsv_image[:,:,0]
            if np.all(hue[pixel_mask]) == 0:
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

    """
    Description:    This function clips the raster image with the provided polygon
                    and saves the clipped image to the output path.
    """
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


def super_clip_interval(directory, year, interval_index, polygon, output_path):
    """
    Description:    This function clips the raster image with the provided polygon
                    and saves the clipped image to the output path.

    Args:       
                directory: str             - The directory containing the raster images
                year: int                  - The agricultural year to study farming activity
                interval_index: int        - The index of the interval within the year eg, 4 for the 4th month,4th fortnight, 4th week
                polygon: shapely.geometry  - The polygon to clip the raster image with
                output_path: str           - The path to save the clipped image to
    
    """

    interval_type = directory.split('/')[-1]
    if year==None:
        try:
            year = os.listdir(directory)[0]         ## Check for data present for any year
        except:
            raise Exception("No quad file for any year")
        
    subdir = os.path.join(directory, str(year), str(interval_index))
    available_quads = os.listdir(os.path.join(directory, str(year)))
    if available_quads == []:
        raise Exception(f"No quad file for year: {year}")
    
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

def remove_padding(raster_path:str, output_path: str):
    """
    Map the polygon cropped farm image to 
    a rectangular representation
    """
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

def brighten_image(image):
    """
    Description:    This function brightens the input image using the OpenCV add function
                    and the PIL ImageEnhance module.

    Args:
                image: np.ndarray    - The input image to be brightened

    Returns:
                brightened_image_pil: np.ndarray   - The brightened image using PIL
    """

    image_pil = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(image_pil)
    image_enhanced = enhancer.enhance(1.5)
    brightened_image= np.array(image_enhanced)

    return brightened_image


