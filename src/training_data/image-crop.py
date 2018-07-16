import configparser
import glob
import json
import rasterio

from rasterio.tools.mask import mask
from helpers.bounding_box import BoundingBox

CONFIG = configparser.ConfigParser()
CONFIG.read('./src/config.ini')

# 1. Open all images from a specific folder
# 2. Loop through all GeoJSON Points and create a bounding box around its longitude and latitude
#    The output image dimension should be 128 x 128px.
#    Therefore we multiply the Ground Sample Distance by 128 to calculate the distance in meters
# 3. Check whether the lng, lat can be found within the bounds of the image
# 4. Create a mask from the polygon GeoJSON Geometry that represents the bounding box
# 5. Apply the mask to the image and store resulting image

def open_geotiff(filename):
    try:
        dataset = rasterio.open(filename)
		# print(f'Image Bands: {dataset.count}')
		# print(f'CRS: {dataset.crs}')
		# print(f'Image Width: {dataset.width}')
		# print(f'Image Height: {dataset.height}')
        print(f'CRS: {dataset.crs}, Bounds: {dataset.bounds}')
        return dataset
    except RuntimeError as err:
        print(err)

def open_geodata():
    with open(CONFIG['Training Data']['geojson_file']) as file:
        geodata = json.load(file)
    return geodata

def process_files_and_coordinates():
	# Loop through geo coordinates
    file_count = 0
    for filename in glob.glob(CONFIG['Training Data']['satellite_imagery_path']):
        feature_count = 0
        file_count += 1
        dataset = open_geotiff(filename)
        for feature in open_geodata()['features']:
            feature_count += 1
            lng = feature['geometry']['coordinates'][0]  # Breitengrad
            lat = feature['geometry']['coordinates'][1]  # Längengrad

            # Get bounding box
            b_box = BoundingBox(lat, lng, dataset.crs)
            geometry = b_box.to_geojson()

            crop_image(dataset, geometry, file_count, feature_count)

# https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html
def crop_image(dataset, geometry, file_count, feature_count):
    try:
        out_image, out_transform = mask(dataset, [geometry], crop=True)
        out_meta = dataset.meta.copy()

        # save the resulting raster
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        img_name = f"{CONFIG['Training Data']['cropped_training_data_path']}/training_img_{file_count}_loc_{feature_count}.tif"
        result_img = rasterio.open(img_name, "w", **out_meta)
        result_img.write(out_image)
        print('.', end='', flush=True)
    except ValueError:
        print('F', end='', flush=True)

process_files_and_coordinates()
