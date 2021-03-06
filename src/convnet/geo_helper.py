import configparser
import json
import os
from time import gmtime, strftime

from helpers.bounding_box import BoundingBox
from osgeo import ogr
import rasterio

CONFIG = configparser.ConfigParser()
CONFIG.read('./src/config.ini')

def bounds_to_polygon(bounds):
    ring = ogr.Geometry(ogr.wkbLinearRing)

    # A rectangle in GeoJSON needs five coordinates: the four corners,
    # and the last one duplicating the first to close the polygon
    ring.AddPoint(bounds.left, bounds.top) # left, top
    ring.AddPoint(bounds.left, bounds.bottom) # left, bottom
    ring.AddPoint(bounds.right, bounds.bottom) # right, bottom
    ring.AddPoint(bounds.right, bounds.top) # right, top
    ring.AddPoint(bounds.left, bounds.top) # left, top

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly

def bounds_to_geojson(bounds, crs):
    epsg = int(crs.to_dict()['init'].split(':')[1])
    polygon = bounds_to_polygon(bounds)
    return BoundingBox.reproject_to_lat_lng_coords(polygon, epsg)

def store_image_bounds(img_path, log_file):
    dataset = rasterio.open(img_path)

    if os.path.exists(log_file):
        with open(log_file) as file:
            data = json.load(file)
    else:
        data = {}
        data["type"] = "FeatureCollection"
        data["features"] = []

    data["features"].append({
        "type": "Feature",
        "geometry": bounds_to_geojson(dataset.bounds, dataset.crs),
        "properties": "Potential ALDFG"
    })

    with open(log_file, 'w') as file:
        json.dump(data, file)
