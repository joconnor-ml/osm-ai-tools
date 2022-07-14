image_size = 224
shard_size = 512
tfrecord_prefix = "gs://osm-object-detector/data/custom_power_plants/tfrecords/shard-"
train_folds = 1
val_folds = 1
num_folds = train_folds + val_folds
