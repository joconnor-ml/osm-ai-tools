import numpy as np
import pandas as pd
import tensorflow as tf
from osm_ai_tools import config

AUTO = tf.data.experimental.AUTOTUNE  # used in tf.data.Dataset API


def get_base_dataset(image_dir, patches):
    image_patches = []
    patch_ids = []
    for image_id, grp in patches.groupby("image_id"):
        image_patches.append(grp[["y_min", "x_min", "y_max", "x_max"]].values)
        patch_ids.append(grp["object_id"].values)

    def patch_gen():
        for coords in image_patches:
            yield coords

    def patch_id_gen():
        for i in patch_ids:
            yield i

    filename_dataset = tf.data.Dataset.from_tensor_slices(
        image_dir + "/" + patches["image_id"].unique() + ".png"
    )
    images = filename_dataset.map(lambda x: tf.io.decode_png(tf.io.read_file(x)))
    bboxes = tf.data.Dataset.from_generator(patch_gen, output_types=tf.float32)
    bbox_ids = tf.data.Dataset.from_generator(patch_id_gen, output_types=tf.int32)
    return tf.data.Dataset.zip((images, bboxes, bbox_ids))


def get_final_dataset(images_and_bboxes, bboxes_per_image):
    # generate positives -- grab crops for each bbox
    def sample_positives(img, bboxes, bbox_ids):
        crops = tf.image.crop_and_resize(
            tf.expand_dims(img, axis=0),
            bboxes,
            box_indices=tf.zeros_like(bboxes[:, 0], dtype=tf.int32),
            crop_size=[config.image_size, config.image_size],
            method="bilinear",
            extrapolation_value=127,
            name=None,
        )
        return tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(crops),
                tf.data.Dataset.from_tensor_slices(bbox_ids),
                tf.data.Dataset.from_tensor_slices([1]).repeat(-1),
            )
        )

    # use random crops for "negatives" -- as long as image size >> object size, this should be OK
    # TODO crop from edges of image to decrease number of false negatives
    def sample_negatives(img, boxes, cls):
        return {
            "image": tf.cast(
                tf.image.random_crop(img, size=[IMAGE_SIZE, IMAGE_SIZE, 3]), np.float32
            ),
            "bbox_id": -1,
            "label": 0,
        }

    positives = images_and_bboxes.flat_map(sample_positives).map(
        lambda img, box_id, cls: {"image": img, "bbox_id": box_id, "label": cls}
    )
    # use `repeat` to balance the data
    negatives = images_and_bboxes.repeat(round(bboxes_per_image)).map(sample_negatives)
    final_dataset = tf.data.experimental.sample_from_datasets([positives, negatives])
    return final_dataset


def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def _int_feature(list_of_ints):  # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def to_tfrecord(img_bytes, label, bbox_id):
    feature = {
        "image": _bytestring_feature([img_bytes]),  # one image in the list
        "label": _int_feature(
            [label]
        ),  # fixed length (1) list of strings, the text label
        "bbox_id": _int_feature([bbox_id]),  # fixed length (1) list of ints
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(input_image_dir, input_bbox_csv, output_tfrecord_path):
    patches = pd.read_csv(input_bbox_csv)

    images_and_bboxes = get_base_dataset(input_image_dir, patches)
    # for balancing positives and negatives:
    bboxes_per_image = patches.shape[0] / patches["image_id"].nunique()

    final_dataset = get_final_dataset(images_and_bboxes, bboxes_per_image)

    def recompress_image(row):
        label = row["label"]
        image = row["image"]
        bbox_id = row["bbox_id"]
        image = tf.cast(image, tf.uint8)
        image = tf.image.encode_jpeg(
            image, optimize_size=True, chroma_downsampling=False
        )
        return image, label, bbox_id

    ds = final_dataset.map(recompress_image).batch(config.shard_size)

    print("Writing TFRecords")
    for shard, (image, label, bbox_id) in enumerate(ds):
        # batch size used as shard size here
        shard_size = image.numpy().shape[0]
        # good practice to have the number of records in the filename
        filename = output_tfrecord_path + "{:02d}-{}.tfrec".format(shard, shard_size)

        print(shard_size)
        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(shard_size):
                print(image.numpy()[i])
                example = to_tfrecord(
                    image.numpy()[i],  # re-compressed image: already a byte string
                    label.numpy()[i],
                    bbox_id.numpy()[i],
                )
                out_file.write(example.SerializeToString())
            print("Wrote file {} containing {} records".format(filename, shard_size))
