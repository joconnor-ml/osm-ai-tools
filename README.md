# OSM AI Tools

Tools for validating and detecting arbitrary objects defined by OSM tags using
high-resolution Mapbox imagery. 

# Installation

```shell
git clone git@github.com:joconnor-ml/osm-ai-tools.git
cd osm-ai-tools
virtualenv --python=python3.8 venv
venv/bin/activate
pip install -r requirements.txt
pip install -e .  # for 'click' command line tools
```

## Data Pipeline

**Note: A Mapbox API token is required to download images. Simplest is using the MAPBOX_ACCESS_TOKEN environment variable.**

Given a set of OSM tags that define a type of object, we can build a training set of images (Mapbox)
and bounding boxes (OSM) that can be used to train object detection and classification models.

See e.g. [examples/cooling_towers/data_pipeline.sh](examples/cooling_towers/data_pipeline.sh)

- `generate_object_location_data` pulls location data using Overpass
- `cluster_objects` can be used to group objects into colocated clusters
to reduce the number of image requests needed.
- `download_images` pulls one image per cluster.
- `generate_bboxes` generates normalised bounding boxes for each object, linking objects and images.

## Image Classification for Tag Validation

Raw OSM annotations are often inconsistent, outdated or wrong. In [mistag_classification.ipynb](example/cooling_towers/mistag_classification.ipynb)
we use a cross-validated image classification pipeline to generate a list of likely
mistagged objects. Objects with a large discrepancy between their out-of-sample
class prediction and their label are likely to be mislabelled.

- Generate positives from images and bounding boxes.
- Generate negatives from random crops. This method relies on images being significantly larger than images to keep mislabelled negatives to a minimum.
- Cross-validation, finetuned RESISC-45 ResNet-50.
- Hand-label candidate mistags (objects with pred_class != true_class)
- Prune mistags from data, retrain model and save.

We boost the performance of classification models using the RESISC-45 aerial imagery
dataset for pretraining. See [train_resisc45_resnet.ipynb](notebooks/train_resisc45_resnet.ipynb).

# TO DO

- [ ] Tag Validation API
- [ ] Object Detection
- [x] Add TPU Compatibility
