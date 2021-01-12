DATA_DIR=data/power_plants
mkdir -p ${DATA_DIR}/images ${DATA_DIR}/tfrecords && \
custom_location_data --input-csv TODO ${DATA_DIR}/object_location_data.csv && \
cluster_objects --input-objects ${DATA_DIR}/object_location_data.csv --output-clusters ${DATA_DIR}/clusters.csv --output-objects ${DATA_DIR}/object_location_data_clustered.csv --max-distance 0.01 && \
download_images --input-csv ${DATA_DIR}/clusters.csv --image-dir ${DATA_DIR}/images --output-csv ${DATA_DIR}/images.csv --zoom 15 && \
generate_bboxes --input-image-csv ${DATA_DIR}/images.csv --input-object-csv ${DATA_DIR}/object_location_data_clustered.csv --output-csv ${DATA_DIR}/bboxes.csv
generate_tfrecords --input-image-dir ${DATA_DIR}/images --input-bbox-csv ${DATA_DIR}/bboxes.csv  --output-tfrecord-path ${DATA_DIR}/tfrecords/shard-
gsutil -m cp -r ${DATA_DIR}/tfrecords/ gs://<project_name>/${DATA_DIR}/tfrecords
