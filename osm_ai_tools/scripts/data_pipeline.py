import json
import os
import subprocess

import click

from . import (
    generate_bboxes,
    generate_object_location_data,
    generate_tfrecords,
    download_images,
    cluster_objects,
)


def main(config_file):
    with open(config_file, "rt") as f:
        conf = json.load(f)
    raw_locations_path = os.path.join(conf["data_dir"], "object_location_data.csv")
    clustered_locations_path = os.path.join(
        conf["data_dir"], "object_location_data_clustered.csv"
    )
    clusters_path = os.path.join(conf["data_dir"], "clusters.csv")
    images_path = os.path.join(conf["data_dir"], "images")
    image_metadata_path = os.path.join(conf["data_dir"], "images.csv")
    bboxes_path = os.path.join(conf["data_dir"], "bboxes.csv")
    tfrecords_path = os.path.join(conf["data_dir"], "tfrecords")

    generate_object_location_data.cli(conf, raw_locations_path, include_tags=True)
    cluster_objects.cli(
        raw_locations_path,
        clusters_path,
        clustered_locations_path,
        max_distance=conf["cluster_size"],
    )
    download_images.cli(
        clusters_path, images_path, image_metadata_path, zoom=conf["zoom"]
    )
    generate_bboxes.cli(image_metadata_path, clustered_locations_path, bboxes_path)
    generate_tfrecords.cli(images_path, bboxes_path, tfrecords_path)
    if conf["gcs_bucket"]:
        subprocess.call(
            [
                "gsutil",
                "-m",
                "cp",
                "-r",
                tfrecords_path,
                os.path.join(conf["gcs_bucket"], tfrecords_path),
            ]
        )


@click.command()
@click.option("--config", help="JSON config file", required=True, type=str)
def cli(config):
    main(config)
