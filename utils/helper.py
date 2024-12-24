from io import BytesIO

import tensorflow as tf
from PIL import Image
import numpy as np

from starlette.datastructures import UploadFile
from keras.src.models.sequential import Sequential

from utils.config import class_mappings


def predict_class(model: Sequential, uploaded_image: UploadFile):
    """
    Predict the class of a given image using the loaded model.

    Parameters:
    model (tf.keras.Model): Loaded model.
    image_path (numpy.ndarray): Image data to be predicted.

    Returns:
    class_name (str): The predicted class name.
    """

    image = uploaded_image.resize((30, 30))
    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    with tf.device("/GPU:0"):
        prediction = np.argmax(model.predict(image), axis=-1)

    return prediction[0], class_mappings[prediction[0]]

