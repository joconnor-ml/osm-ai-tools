import math

import click
import pandas as pd

from . import download_images


def getPointLatLng(x, y, lat, lng, size_x, size_y, zoom):
    parallelMultiplier = math.cos(lat * math.pi / 180)
    degreesPerPixelX = 360 / math.pow(2, zoom + 8)
    degreesPerPixelY = 360 / math.pow(2, zoom + 8) * parallelMultiplier
    pointLat = lat - degreesPerPixelY * (y - size_y / 2)
    pointLng = lng + degreesPerPixelX * (x - size_x / 2)

    return (pointLat, pointLng)


image_size = 0.01


def get_patch(row, padding=0.06):
    ne = getPointLatLng(
        row.size_x,
        0,
        row.center_lat_image,
        row.center_lon_image,
        row.size_x,
        row.size_y,
        row.zoom,
    )
    nw = getPointLatLng(
        0,
        0,
        row.center_lat_image,
        row.center_lon_image,
        row.size_x,
        row.size_y,
        row.zoom,
    )
    se = getPointLatLng(
        row.size_x,
        row.size_y,
        row.center_lat_image,
        row.center_lon_image,
        row.size_x,
        row.size_y,
        row.zoom,
    )
    size_lat = ne[0] - se[0]
    size_lon = ne[1] - nw[1]
    return pd.Series(
        dict(
            y_min=(0.5 - (row.min_lat - row.center_lat_image) / size_lat)
            + padding,  # add a small buffer
            y_max=(0.5 - (row.max_lat - row.center_lat_image) / size_lat) - padding,
            x_min=(0.5 + (row.min_lon - row.center_lon_image) / size_lon) - padding,
            x_max=(0.5 + (row.max_lon - row.center_lon_image) / size_lon) + padding,
            object_id=row.object_id,
        )
    )


@click.command()
@click.option(
    "--input-image-csv",
    help="CSV of image location, zoom and size information",
    required=True,
    type=str,
)
@click.option(
    "--input-object-csv",
    help="CSV of object location data: should be called object_location_data_clustered.csv",
    required=True,
    type=str,
)
@click.option("--output-csv", help="Path to output bbox CSV", required=True, type=str)
def cli(input_image_csv, input_object_csv, output_csv):
    image_df = pd.read_csv(input_image_csv)
    image_df["image_id"] = image_df.apply(
        lambda row: download_images.get_image_id(
            row.center_lat, row.center_lon, row.zoom, row.size_x, row.size_y
        ),
        axis=1,
    )
    object_df = pd.read_csv(input_object_csv).merge(
        image_df, how="inner", on="cluster_id", suffixes=("", "_image")
    )

    bboxes = object_df.apply(get_patch, axis=1)
    bboxes["object_id"] = bboxes["object_id"].astype(int)
    bboxes["image_id"] = object_df["image_id"]
    bboxes.to_csv(output_csv, index=False)
