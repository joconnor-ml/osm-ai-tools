import os
import subprocess as sp

import backoff
import pandas as pd
import tensorflow as tf
from loguru import logger
from tqdm import tqdm


def get_image_id(lat: float, lon: float, zoom: int, size_x: int, size_y: int) -> str:
    return f"lat_{lat:.5f}_lon_{lon:.5f}_zoom_{zoom}_{size_x}x{size_y}"


def image_readable(path):
    tf.io.decode_png(tf.io.read_file(path))
    return True


@backoff.on_exception(backoff.expo, (sp.CalledProcessError, RuntimeError), max_tries=3)
def download_image(
    lat: float, lon: float, zoom: int, size_x: int, size_y: int, filename: str
) -> None:
    if os.path.exists(filename):
        logger.debug(f"file {filename} exists: skipping")
        return
    call = [
        "mapbox",
        "staticmap",
        "--lon",
        str(lon),
        "--lat",
        str(lat),
        "--zoom",
        str(zoom),
        "--size",
        str(size_x),
        str(size_y),
        "mapbox.satellite",
        filename,
    ]
    sp.check_call(call)  # throws if call fails
    # mapbox can silently fail and return a size-zero image. Check for this
    if not image_readable(filename):
        raise RuntimeError("image download failed")


def main(
    input_csv: str, image_dir: str, output_csv: str, image_size: int, zoom: int
) -> None:
    image_requests = pd.read_csv(input_csv)
    image_requests["zoom"] = zoom
    image_requests["size_x"] = image_size
    image_requests["size_y"] = image_size
    for col in ["zoom", "size_x", "size_y"]:
        image_requests[col] = image_requests[col].astype("Int64")

    image_ids = []
    image_status = []
    for i, row in tqdm(image_requests.iterrows(), total=image_requests.shape[0]):
        image_id = get_image_id(
            row.center_lat, row.center_lon, row.zoom, row.size_x, row.size_y
        )
        filename = os.path.join(image_dir, f"{image_id}.png")
        try:
            download_image(
                row.center_lat,
                row.center_lon,
                row.zoom,
                row.size_x,
                row.size_y,
                filename,
            )
            image_status.append(True)
        except Exception as e:
            logger.exception(e)
            # note down failure
            image_status.append(False)

        image_ids.append(image_id)
    image_requests["image_id"] = image_ids
    image_requests["image_downloaded"] = image_status
    num_failures = (image_requests["image_downloaded"] == False).sum()
    logger.info(f"{num_failures} images failed to download")
    image_requests.to_csv(output_csv, index=False, float_format="%.5f")
