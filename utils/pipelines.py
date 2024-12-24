from io import BytesIO

from PIL import Image
from utils.helper import predict_class
from keras.src.models.sequential import Sequential
from starlette.datastructures import UploadFile

def predict_class_from_uploaded_image(model: Sequential, uploaded_image: UploadFile):
    """
    Predict the class of a given image using the loaded model.

    Parameters:
    model (tf.keras.Model): Loaded model.
    image_path (numpy.ndarray): Image data to be predicted.

    Returns:
    class_name (str): The predicted class name.
    """
    image_bytes = uploaded_image.file.read()
    image = Image.open(BytesIO(image_bytes))
    predicted_class, prediction = predict_class(model=model, uploaded_image=image)
    print(f"Predicted class: {predicted_class}, Prediction: {prediction}")
    return prediction