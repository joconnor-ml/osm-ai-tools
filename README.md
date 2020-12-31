# OSM AI Tools

Tools for validating and detecting arbitrary objects defined by OSM tags using
high-resolution Mapbox imagery.

## Object Detection

Given a set of OSM tags that define a type of object, we can build a training set of images (Mapbox)
and bounding boxes (OSM) that can be used to train an object detection model. This
model can then be applied to new areas to search for new, untagged objects.

- `generate_object_location_data` pulls location data using Overpass
- `cluster_objects` can be used to group objects into colocated clusters
to reduce the number of image requests needed.
- `download_images` pulls one image per cluster

We then make use of the Tensorflow Object Detection API to build an object detector.

## Tag Validation (Classification)

Raw OSM annotations are often inconsistent, outdated or wrong.
This can lead to difficulty in training an object detection model.

Here we sample images from the image clusters defined above to build a
classification model. Applying cross-validation allows us to detect the
annotations that the model finds most surprising, which are likely to be
mistakes. This list can be used to correct the upstream OSM database, or
to prune mislabelled objects from a local image detection dataset.

For classification models, we make use of the RESISC-45 aerial imagery dataset
and Google AI's Big Transfer models and methodology.
