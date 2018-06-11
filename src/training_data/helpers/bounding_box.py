import json
from osgeo import ogr
from osgeo import osr

# Distance between the centres of two neighbouring pixel in the image on the ground
GROUND_SAMPLE_DISTANCE = 4.0

class BoundingBox:
    def __init__(self, lat, lng):
        self.calculate(lat, lng)

    def calculate(self, lat, lng, size_in_pixel=32):
        diameter = size_in_pixel * GROUND_SAMPLE_DISTANCE

        utm_coords = self.reproject_to_utm_coords(lat, lng)
        easting = utm_coords['coordinates'][0]
        northing = utm_coords['coordinates'][1]

        self.easting_min = easting - diameter/2
        self.easting_max = easting + diameter/2
        self.northing_min = northing - diameter/2
        self.northing_max = northing + diameter/2

    def to_polygon(self):
        ring = ogr.Geometry(ogr.wkbLinearRing)

        # A rectangle in GeoJSON needs five coordinates: the four corners,
        # and the last one duplicating the first to close the polygon
        ring.AddPoint(self.easting_min, self.northing_max) # left, top
        ring.AddPoint(self.easting_min, self.northing_min) # left, bottom
        ring.AddPoint(self.easting_max, self.northing_min) # right, bottom
        ring.AddPoint(self.easting_max, self.northing_max) # right, top
        ring.AddPoint(self.easting_min, self.northing_max) # left, top

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        return poly

    def to_geojson(self):
        poly = self.to_polygon()
        return json.loads(poly.ExportToJson())

    # Reproject to UTM-Coordinates (WGS84)
    @staticmethod
    def reproject_to_utm_coords(lat, lng):
        source = osr.SpatialReference()
        # Google Earth CRS
        source.ImportFromEPSG(4326)

        target = osr.SpatialReference()
        # Planet Labs CRS
        target.ImportFromEPSG(32635)

        transform = osr.CoordinateTransformation(source, target)

        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(lng, lat)
        point.Transform(transform)
        return json.loads(point.ExportToJson())

    @staticmethod
    def reproject_to_lat_lng_coords(geometry):
        source = osr.SpatialReference()
        # Google Earth CRS
        source.ImportFromEPSG(32635)

        target = osr.SpatialReference()
        # Planet Labs CRS
        target.ImportFromEPSG(4326)

        transform = osr.CoordinateTransformation(source, target)
        geometry.Transform(transform)
        return json.loads(geometry.ExportToJson())
