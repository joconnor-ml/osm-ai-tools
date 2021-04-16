DATA_DIR=data/gems_plants
mkdir -p ${DATA_DIR}/images ${DATA_DIR}/tfrecords && \
download_images --input-csv ${DATA_DIR}/clusters.csv --image-dir ${DATA_DIR}/images --output-csv ${DATA_DIR}/images.csv --zoom 15 && \
gsutil -m cp -r ${DATA_DIR}/tfrecords/ gs://<project_name>/${DATA_DIR}/tfrecords
