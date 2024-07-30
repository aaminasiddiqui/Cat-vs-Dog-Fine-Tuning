from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array, load_img

#Data augmentation

train_datagen=ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen=ImageDataGenerator(rescale=1./255)

#Generators
batch_size=12
train_generator=train_datagen.flow_from_directory(
    '/content/drive/MyDrive/CATS_DOGS/train',
    target_size=(150,150),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator=test_datagen.flow_from_directory(
    '/content/drive/MyDrive/CATS_DOGS/test',
    target_size=(150,150),
    batch_size=batch_size,
    class_mode='binary'
)

from keras.applications.vgg16 import VGG16

model=VGG16()
model.summary()

from tensorflow.keras import Model

from tensorflow.keras.utils import plot_model

plot_model(model)

conv_base = VGG16(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
model=Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

conv_base.trainable=True
for layer in conv_base.layers:
    if layer.name=='block5_conv1':
        layer.trainable=True
    else:
        layer.trainable=False
for layer in conv_base.layers:
    print('layer.name',layer.trainable)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

results=model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

from keras.models import load_model

model.save('CatDogFine.h5')
model1=load_model('CatDogFine.h5')

import matplotlib.pyplot as plt

plt.plot(results.history['accuracy'],color='red',label='train')
plt.plot(results.history['val_accuracy'],color='blue',label='test')
plt.legend()
plt.show()

from tensorflow.keras.preprocessing import image
import numpy as np

cat_img=image.load_img('/content/drive/MyDrive/cat.jpg',target_size=(150,150))
cat_img=image.img_to_array(cat_img)
cat_img=np.expand_dims(cat_img,axis=0)
cat_img=cat_img/255

prediction=model1.predict(cat_img)
classification=np.argmax(model1.predict(cat_img), axis=-1)

classification