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
    custom_location_data,
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

    if conf["osm_tags"]:
        generate_object_location_data.cli(
            query_config=conf, output_csv=raw_locations_path, include_tags=True
        )
    elif conf["custom_locations"]:
        custom_location_data.cli(
            conf["custom_locations"],
            id_col="gems_plant_id",
            lat_col="lat",
            lon_col="lon",
            object_size=0.01,
            object_class="power_plant",
            output_csv=raw_locations_path,
        )
    cluster_objects.cli(
        input_objects=raw_locations_path,
        output_clusters=clusters_path,
        output_objects=clustered_locations_path,
        max_distance=conf["cluster_size"],
    )
    download_images.cli(
        input_csv=clusters_path,
        image_dir=images_path,
        output_csv=image_metadata_path,
        zoom=conf["zoom"],
    )
    generate_bboxes.cli(
        input_image_csv=image_metadata_path,
        input_object_csv=clustered_locations_path,
        output_csv=bboxes_path,
    )
    generate_tfrecords.cli(
        input_image_dir=images_path,
        input_bbox_csv=bboxes_path,
        output_tfrecord_path=tfrecords_path,
    )
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
