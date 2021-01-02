DATA_DIR=data/cooling_towers
mkdir -p ${DATA_DIR}/images && \
generate_object_location_data --query-config example/cooling_towers/query.json --output-csv ${DATA_DIR}/object_location_data.csv && \
cluster_objects --input-objects ${DATA_DIR}/object_location_data.csv --output-clusters ${DATA_DIR}/clusters.csv --output-objects ${DATA_DIR}/object_location_data_clustered.csv && \
download_images --input-csv ${DATA_DIR}/clusters.csv --image-dir ${DATA_DIR}/images --output-csv ${DATA_DIR}/images.csv && \
generate_bboxes --input-csv ${DATA_DIR}/images.csv --output-csv ${DATA_DIR}/bboxes.csv