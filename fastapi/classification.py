from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils

input_shape = (224, 224)


def load_model2():
    model = tf.keras.applications.VGG19(input_shape)
    return model

model2 = load_model2()

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image


def preprocess(image: Image.Image):
    image = image.resize(input_shape)
    image = np.asfarray(image)
    image = image/127.5 - 1.0
    image = np.expand_dims(image,0)

    return image


def predicti(image: np.ndarray):
    predictions = model2.predict(image)
    predictions = imagenet_utils.decode_predictions(predictions)[0][1][1]
    return predictions
