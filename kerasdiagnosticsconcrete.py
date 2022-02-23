import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense

os.listdir('concrete_data_week4')
dataset_dir = './concrete_data_week4'
train_set = dataset_dir + '/_MACOSX'
val_set = dataset_dir + '/concrete_data_week4'

data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

image_resize = 224

train_generator = data_generator.flow_from_directory(
    dataset_dir,
    target_size=(image_resize, image_resize),
    batch_size=100,
    class_mode='categorical',
    seed=24
    )

validation_generator = data_generator.flow_from_directory(
    dataset_dir,
    target_size=(image_resize, image_resize),
    batch_size=100,
    class_mode='categorical',
    seed=24
    )

model = Sequential()

model.add(VGG16(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))

model.add(Dense(2, activation='softmax'))
model.layers[0].trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(validation_generator)
num_epochs = 2

fit_history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
)
