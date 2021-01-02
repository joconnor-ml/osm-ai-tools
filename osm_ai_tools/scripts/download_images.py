import os
import subprocess as sp

import click
import pandas as pd
from tqdm import tqdm


def get_image_id(lat: float, lon: float, zoom: int, size_x: int, size_y: int) -> str:
    return f"lat_{lat:.5f}_lon_{lon:.5f}_zoom_{zoom}_{size_x}x{size_y}"


def download_image(lat, lon, zoom, size_x, size_y, filename):
    call = ["mapbox", "staticmap",
            "--lon", str(lon),
            "--lat", str(lat),
            "--zoom", str(zoom),
            "--size", str(size_x), str(size_y),
            "mapbox.satellite", filename]
    sp.call(call)


def download_all_images(location_file: str, image_dir: str, output_csv: str, image_size: int, zoom: int) -> None:
    image_requests = pd.read_csv(location_file)
    image_requests["zoom"] = zoom
    image_requests["size_x"] = image_size
    image_requests["size_y"] = image_size
    for col in ["zoom", "size_x", "size_y"]:
        image_requests[col] = image_requests[col].astype("Int64")

    image_ids = []
    for i, row in tqdm(image_requests.iterrows(), total=image_requests.shape[0]):
        image_id = get_image_id(row)
        filename = os.path.join(image_dir, f"{image_id}.png")
        download_image(row.center_lat, row.center_lon, row.zoom, row.size_x, row.size_y, filename)
        image_ids.append(image_id)
    image_requests["image_id"] = image_ids
    image_requests.to_csv(output_csv, index=False, float_format="%.5f")


@click.command()
@click.option('--input-csv', help='CSV of lat, lon, zoom specifying images to download', required=True, type=str)
@click.option('--image-dir', help='Path to output directory', required=True, type=str)
@click.option('--output-csv', help='Path to output CSV file containing image IDs', required=True, type=str)
@click.option('--image-size', help='image size in pixels, max=1280', default=1280, type=int)
@click.option('--zoom', help='image zoom level', default=17, type=int)
def cli(input_csv, image_dir, output_csv, image_size, zoom):
    download_all_images(input_csv, image_dir, output_csv, image_size, zoom)
