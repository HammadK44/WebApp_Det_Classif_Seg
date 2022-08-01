from sqlalchemy import false
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
import streamlit as st
from datetime import datetime
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


st.title("Classification Batch Prediction with Evaluation Metrics")


def predict(IMAGE_SIZE, train_path, test_path):
    vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    vgg.input


    for layer in vgg.layers:
        layer.trainable = False

    folders = glob('./dataset/train/*')



    x = Flatten()(vgg.output)
    prediction = Dense(len(folders), activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction)
    model.summary()


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




    train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')


    test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')


    train_set = train_datagen.flow_from_directory(train_path,
                                                        target_size = (224, 224),
                                                        batch_size = 8,
                                                        class_mode = 'categorical')




    test_set = test_datagen.flow_from_directory(test_path,
                                                    target_size = (224, 224),
                                                    batch_size = 8,
                                                    class_mode = 'categorical')



    model_history=model.fit_generator(
        train_set,
        validation_data=test_set,
        epochs=3,
        steps_per_epoch=5,
        validation_steps=8,verbose=2)


    accuracy = model_history.history['accuracy']
    loss = model_history.history['loss']
    precision = model_history.history['recall']
    recall = model_history.history['precision']
    f1score = model_history.history['F1_score']

    accuracy = accuracy[-1]
    loss = loss[-1]
    precision = precision[-1]
    recall = recall[-1]
    f1score = f1score[-1]

    return accuracy, loss, precision, recall, f1score

    


IMAGE_SIZE = [224, 224]

train_path = './dataset/train'
test_path = './dataset/test'


st.write("Model Running")



#progress = st.progress(0)
#start = datetime.now()



accuracy, loss, precision, recall, f1score = predict(IMAGE_SIZE, train_path, test_path)

#duration = datetime.now() - start
st.write(f'Accuracy =', accuracy)
st.write(f'Loss =', loss)
st.write(f'Precision =', precision)
st.write(f'Recall =', recall)
st.write(f'F1 Score =', f1score)