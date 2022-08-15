import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def predict(test_path):
    IMAGE_SIZE = [224, 224]
    vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    vgg.input


    for layer in vgg.layers:
        layer.trainable = False
    
    #breakpoint()

    folders = glob('./dataset/test/*')



    x = Flatten()(vgg.output)
    prediction = Dense(len(folders), activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction)
    #model.summary()


    from tensorflow.keras import optimizers
    import tensorflow.keras.backend as K


    def F1_score(y_true, y_pred):                                     
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val


    adam = optimizers.Adam()
    model.compile(loss='binary_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy',tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1_score])

    test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    test_set = test_datagen.flow_from_directory(test_path,
                                                    target_size = (224, 224),
                                                    class_mode = 'categorical')



    numberofimages = test_set.samples
    ev = model.evaluate_generator(test_set, numberofimages)

    
    loss = ev[1]
    accuracy = ev[0]
    precision = ev[2]
    recall = ev[3]
    f1score = ev[4]

    return loss, accuracy, precision, recall, f1score
