import tensorflow as tf

from .scripts.download_images import download_image, get_image_id


def predict(model, lat, lon, zoom, size):
    """Run inference on a certain lat/lon point. Ensure that the object of interest is well centred and contained
     with the given image size and zoom level. All images will be scaled (not cropped) to the model input size."""
    image_filename = f"/tmp/{get_image_id(lat, lon, zoom, size, size)}"
    download_image(lat, lon, zoom, size, size, image_filename)
    img = tf.io.decode_png(tf.io.read_file(f"data/images/{image_filename}.png"))
    img = tf.image.resize(img, [224, 224, 3])
    return model.predict(tf.expand_dims(img, axis=0)).item()
