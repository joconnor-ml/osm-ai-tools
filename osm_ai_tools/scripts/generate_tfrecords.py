import tensorflow as tf

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

import pandas as pd

image_df = pd.read_csv("data/images.csv")
object_df = pd.read_csv("data/object_location_data_clustered.csv").merge(
    image_df, how="left", on="cluster_id", suffixes=("", "_image")
)

# !/usr/bin/python
import math

w = 1280
h = 1280
zoom = 17


def getPointLatLng(x, y, lat, lng):
    parallelMultiplier = math.cos(lat * math.pi / 180)
    degreesPerPixelX = 360 / math.pow(2, zoom + 8)
    degreesPerPixelY = 360 / math.pow(2, zoom + 8) * parallelMultiplier
    pointLat = lat - degreesPerPixelY * (y - h / 2)
    pointLng = lng + degreesPerPixelX * (x - w / 2)

    return (pointLat, pointLng)


image_size = 0.01


def get_patch(row):
    ne = getPointLatLng(w, 0, row.center_lat_image, row.center_lon_image)
    sw = getPointLatLng(0, h, row.center_lat_image, row.center_lon_image)
    nw = getPointLatLng(0, 0, row.center_lat_image, row.center_lon_image)
    se = getPointLatLng(w, h, row.center_lat_image, row.center_lon_image)
    size_lat = ne[0] - se[0]
    size_lon = ne[1] - nw[1]
    return pd.Series(dict(
        y_min=0.5 - (row.min_lat - row.center_lat_image) / size_lat,
        y_max=0.5 - (row.max_lat - row.center_lat_image) / size_lat,
        x_min=0.5 + (row.min_lon - row.center_lon_image) / size_lon,
        x_max=0.5 + (row.max_lon - row.center_lon_image) / size_lon,
    ))


patches = object_df.apply(get_patch, axis=1)
patches["cluster_id"] = object_df["cluster_id"]

image_patches = []
for cid in image_df.cluster_id:
    image_patches.append(patches.loc[patches["cluster_id"] == cid, ["y_min", "x_min", "y_max", "x_max"]].values)


def create_tf_example(row, patches):
    # TODO(user): Populate the following variables from your example.
    height = 1024  # Image height
    width = 1024  # Image width
    filename = f"data/images/{image_id}.png"  # Filename of the image. Empty if image is not from file
    encoded_image_data = None  # Encoded image bytes
    image_format = "png"  # b'jpeg' or b'png'

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # TODO(user): Write code to read in your dataset to examples variable

    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
