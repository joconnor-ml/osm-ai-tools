import json
import os
import subprocess

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

    os.makedirs(conf["data_dir"], exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(tfrecords_path, exist_ok=True)

    if "osm_tags" in conf:
        generate_object_location_data.main(
            query_config=conf, output_csv=raw_locations_path, include_tags=True
        )
    elif "custom_locations" in conf:
        custom_location_data.main(
            conf["custom_locations"]["file"],
            id_col=conf["custom_locations"]["id_col"],
            lat_col=conf["custom_locations"]["lat_col"],
            lon_col=conf["custom_locations"]["lon_col"],
            object_size=conf["cluster_size"],
            object_class=conf["object_class"],
            output_csv=raw_locations_path,
        )
    else:
        raise RuntimeError(
            "One of 'osm_tags' or 'custom_locations' required in config file."
        )
    cluster_objects.main(
        input_objects=raw_locations_path,
        output_clusters=clusters_path,
        output_objects=clustered_locations_path,
        max_distance=conf["cluster_size"],
    )
    download_images.main(
        input_csv=clusters_path,
        image_dir=images_path,
        output_csv=image_metadata_path,
        image_size=conf["image_download_size"],
        zoom=conf["zoom"],
    )
    generate_bboxes.main(
        input_image_csv=image_metadata_path,
        input_object_csv=clustered_locations_path,
        output_csv=bboxes_path,
    )
    generate_tfrecords.main(
        input_image_dir=images_path,
        input_bbox_csv=bboxes_path,
        output_tfrecord_path=tfrecords_path,
    )
    if "gcs_bucket" in conf:
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    args = parser.parse_args()
    main(args.config)
