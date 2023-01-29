import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from io import BytesIO


def fast_model(content_image: Image,
               style_image: Image) -> BytesIO:
    """
     Initialize style transfer process
    and get bytes of result image

    Parameters
    ----------
    content_image (Image):
        Content image in PIL format
    style_image (Image):
        Style image in PIL format
    Returns
    -------
        Bytes of result image after style transfer process
    """
    content_image = np.asarray(content_image)  # transform PIL format to numpy
    style_image = np.asarray(style_image)  # transform PIL format to numpy
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.  # transform array to float type
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.  # transform array to float type
    style_image = tf.image.resize(style_image, (512, 512))  # resize style
    model = hub.load('./fast_model/saved_model')  # load pretrained model
    outputs = model(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]  # get stylized picture
    image = (stylized_image.numpy().squeeze() * 255).astype('uint8')  # remove first dimension and return to RGB format
    image = Image.fromarray(image)  # transform numpy array to PIL format
    bio = BytesIO()
    bio.name = 'image.jpeg'
    image.save(bio, 'JPEG')  # save bytes of stylized image
    bio.seek(0)
    return bio
