import math

import pandas as pd



def getPointLatLng(x, y, lat, lng, size_x, size_y, zoom):
    parallelMultiplier = math.cos(lat * math.pi / 180)
    degreesPerPixelX = 360 / math.pow(2, zoom + 8)
    degreesPerPixelY = 360 / math.pow(2, zoom + 8) * parallelMultiplier
    pointLat = lat - degreesPerPixelY * (y - size_y / 2)
    pointLng = lng + degreesPerPixelX * (x - size_x / 2)

    return (pointLat, pointLng)


image_size = 0.01


def get_patch(row, padding=0.06):
    ne = getPointLatLng(row.size_x, 0, row.center_lat_image, row.center_lon_image, row.size_x, row.size_y, row.zoom)
    nw = getPointLatLng(0, 0, row.center_lat_image, row.center_lon_image, row.size_x, row.size_y, row.zoom)
    se = getPointLatLng(row.size_x, row.size_y, row.center_lat_image, row.center_lon_image, row.size_x, row.size_y, row.zoom)
    size_lat = ne[0] - se[0]
    size_lon = ne[1] - nw[1]
    return pd.Series(dict(
        y_min=(0.5 - (row.min_lat - row.center_lat_image) / size_lat) + padding,  # add a small buffer
        y_max=(0.5 - (row.max_lat - row.center_lat_image) / size_lat) - padding,
        x_min=(0.5 + (row.min_lon - row.center_lon_image) / size_lon) - padding,
        x_max=(0.5 + (row.max_lon - row.center_lon_image) / size_lon) + padding,
        osm_id=row.osm_id
    ))


image_df = pd.read_csv("data/images.csv")
object_df = pd.read_csv("data/object_location_data_clustered.csv").merge(
    image_df, how="left", on="cluster_id", suffixes=("", "_image")
).merge(image_df, on="cluster_id")


bboxes = object_df.apply(get_patch, axis=1)
bboxes["image_id"] = object_df["image_id"]
bboxes["osm_id"] = bboxes["osm_id"].astype(int)
bboxes.to_csv("data/bboxes.csv")
