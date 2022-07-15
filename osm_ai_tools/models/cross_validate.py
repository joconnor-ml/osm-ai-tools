import numpy as np
import pandas as pd
import tensorflow as tf
from osm_ai_tools import config

AUTO = tf.data.experimental.AUTOTUNE  # used in tf.data.Dataset API
BATCH_SIZE = 128


def get_model():
    module = tf.keras.models.load_model(
        "gs://osm-object-detector/pretrained_models/resisc_224px_rgb_resnet50"
    )
    module.trainable = True
    module.summary()

    images = tf.keras.layers.Input((config.image_size, config.image_size, 3))
    features = module(images)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(0.5)(features)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(features)
    model = tf.keras.Model(inputs=images, outputs=output)

    lr = 0.003 * BATCH_SIZE / 512

    # Decay learning rate by a factor of 10 at SCHEDULE_BOUNDARIES.
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[
            int(50 * BATCH_SIZE / 512),
            int(75 * BATCH_SIZE / 512),
            int(100 * BATCH_SIZE / 512),
        ],
        values=[lr, lr * 0.1, lr * 0.001, lr * 0.0001],
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

    model.compile(
        optimizer=optimizer,
        # use label smoothing since we know quite a few labels will be wrong
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=["acc"],
    )
    return model


def read_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature(
            [], tf.string
        ),  # tf.string = bytestring (not text string)
        "label": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
        "bbox_id": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    # FixedLenFeature fields are now ready to use: exmple['size']
    # VarLenFeature fields require additional sparse_to_dense decoding

    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = tf.reshape(image, [config.image_size, config.shard_size, 3])
    image = tf.cast(image, tf.float32)

    return {"image": image, "label": example["label"], "bbox_id": example["bbox_id"]}


def to_keras(row):
    return row["image"], row["label"]


augmentor = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip(
            "horizontal_and_vertical"
        ),
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
        tf.keras.layers.experimental.preprocessing.Resizing(256, 256),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2),
        tf.keras.layers.experimental.preprocessing.RandomRotation(2 * math.pi),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.25),
        tf.keras.layers.experimental.preprocessing.CenterCrop(224, 224),
    ]
)


def augment(image_batch, label_batch):
    return augmentor.call(image_batch), label_batch


def main(output_dir):
    # read from TFRecords. For optimal performance, read from multiple
    # TFRecord files at once and set the option experimental_deterministic = False
    # to allow order-altering optimizations.

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    filenames = tf.io.gfile.glob(config.tfrecord_prefix + "*.tfrec")
    # split folds at this stage for quicker loading.

    # split by taking every 'i'th file for i in num folds, i.e. [1, 2, 3, 1, 2, 3] etc.
    fold_files = np.array(
        filenames[i :: config.num_folds] for i in range(config.num_folds)
    )

    def tf_dataset_from_files(filenames):
        ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
        ds = ds.with_options(option_no_order)
        ds = ds.map(read_tfrecord, num_parallel_calls=AUTO)
        ds = ds.shuffle(len(filenames))
        return ds

    def get_pred_df(model, val_ds):
        pred_dfs = []
        for row in val_ds.batch(BATCH_SIZE).take(-1):
            preds = model.predict(row["image"])
            pred_dfs.append(
                pd.DataFrame(
                    {
                        "pred": preds.flatten(),
                        "label": row["label"].numpy(),
                        "osm_id": row["bbox_id"].numpy(),
                    }
                )
            )
        return pd.concat(pred_dfs).reset_index(drop=True)

    pred_dfs = []
    for i in range(config.num_folds):
        train_files = np.roll(fold_files, i)[: config.train_folds]
        val_files = np.roll(fold_files, i)[config.train_folds :]
        train_ds = tf_dataset_from_files(train_files)
        val_ds = tf_dataset_from_files(val_files)

        model = get_model()
        # note: TPU training requires drop_remainder=True to keep batch sizes constant
        model.fit(
            train_ds.shuffle(500)
            .map(to_keras)
            .batch(BATCH_SIZE, drop_remainder=True)
            .map(augment, num_parallel_calls=AUTO)
            .prefetch(-1),
            validation_data=val_ds.map(to_keras).batch(BATCH_SIZE).prefetch(-1),
            epochs=3,
        )
        pred_dfs.append(get_pred_df(model, val_ds))

    pd.concat(pred_dfs).to_csv(output_dir)
