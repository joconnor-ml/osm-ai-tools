DATA_DIR=data/power_plants
mkdir -p ${DATA_DIR}/images ${DATA_DIR}/tfrecords && \
generate_object_location_data --query-config examples/power_plants/query.json --output-csv ${DATA_DIR}/object_location_data.csv --include-tags && \
cluster_objects --input-objects ${DATA_DIR}/object_location_data.csv --output-clusters ${DATA_DIR}/clusters.csv --output-objects ${DATA_DIR}/object_location_data_clustered.csv && \
download_images --input-csv ${DATA_DIR}/clusters.csv --image-dir ${DATA_DIR}/images --output-csv ${DATA_DIR}/images.csv && \
generate_bboxes --input-image-csv ${DATA_DIR}/images.csv --input-object-csv ${DATA_DIR}/object_location_data_clustered.csv --output-csv ${DATA_DIR}/bboxes.csv
generate_tfrecords --input-image-dir ${DATA_DIR}/images --input-bbox-csv ${DATA_DIR}/bboxes.csv  --output-tfrecord-path ${DATA_DIR}/tfrecords/shard-
