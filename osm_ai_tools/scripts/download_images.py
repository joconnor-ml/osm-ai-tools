import os
import subprocess as sp

import click
import pandas as pd
from tqdm import tqdm


def get_image_id(row: pd.Series) -> str:
    return f"lat_{row.lat:.5f}_lon_{row.lon:.5f}_zoom_{row.zoom}_{row.size_x}x{row.size_y}"


def download_images(location_file: str, image_dir: str, output_csv: str) -> None:
    image_requests = pd.read_csv(location_file)
    for col in ["zoom", "size_x", "size_y"]:
        image_requests[col] = image_requests[col].astype("Int64")

    image_ids = []
    for i, row in tqdm(image_requests.iterrows(), total=image_requests.shape[0]):
        image_id = get_image_id(row)
        filename = os.path.join(image_dir, f"{image_id}.png")
        call = ["mapbox", "staticmap",
                "--lon", str(row.lon),
                "--lat", str(row.lat),
                "--zoom", str(row.zoom),
                "--size", str(row.size_x), str(row.size_y),
                "mapbox.satellite", filename]
        sp.call(call)
        image_ids.append(image_id)
    image_requests["image_id"] = image_ids
    image_requests.to_csv(output_csv, index=False, float_format="%.5f")


@click.command()
@click.option('--input-csv', help='CSV of lat, lon, zoom specifying images to download', required=True, type=str)
@click.option('--image-dir', help='Path to output directory', required=True, type=str)
@click.option('--output-csv', help='Path to output CSV file containing image IDs', required=True, type=str)
def cli(input_csv, image_dir, output_csv):
    download_images(input_csv, image_dir, output_csv)
