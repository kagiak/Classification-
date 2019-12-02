import numpy as np
import os
import json
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import csv
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
import keras

image_size = 299#224
trainFile = 'food-11/train.json'
valFile = 'food-11/val.json'
batch_size = 4
categories = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']

train_datagen = ImageDataGenerator(featurewise_center=True,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)

def openJson(file):
    with open(file) as File:
        dict = json.load(File)
    return dict

def train_generator():
    with open(trainFile) as trainfile:
        dict_train = json.load(trainfile)

    train = pd.DataFrame.from_dict(dict_train, orient='index')
    train.reset_index(level=0, inplace=True)
    train.columns = ['Id', 'Ingredients', 'Binary']
    nb_samples = len(train)

    while True:
        for start in range(0, nb_samples, batch_size):
            train_image =[]
            y_batch = []
            end = min(start + batch_size, nb_samples)
            for i in range(start, end):
                img = image.load_img('food-11/training/' + train['Id'][i], target_size=(image_size, image_size, 3))
                img = image.img_to_array(img)
                img = img / 255
                img = 2*((img - np.min(img)) / (np.max(img) - np.min(img))) - 1
                train_image.append(img)

                y_batch.append(train['Binary'][i])

            # return np.array(train_image), np.array(y_batch)
            yield (np.array(train_image), np.array(y_batch))

def val_generator():
    with open(valFile) as valfile:
        dict_val = json.load(valfile)

    val = pd.DataFrame.from_dict(dict_val, orient='index')
    val.reset_index(level=0, inplace=True)
    val.columns = ['Id', 'Ingredients', 'Binary']

    nb_samples = len(val)

    while True:
        for start in range(0, nb_samples, batch_size):
            val_image = []
            y_batch = []
            end = min(start + batch_size, nb_samples)
            for i in range(start, end):
                img = image.load_img('food-11/validation/' + val['Id'][i], target_size=(image_size, image_size, 3))
                img = image.img_to_array(img)
                img = img / 255
                img = 2 * ((img - np.min(img)) / (np.max(img) - np.min(img))) - 1
                val_image.append(img)

                y_batch.append(val['Binary'][i])

            yield (np.array(val_image), np.array(y_batch))
            # return np.array(val_image), np.array(y_batch)

nb_train_samples =  len(openJson(trainFile))
nb_valid_samples = len(openJson(valFile))

print("TRAIN LEN", nb_train_samples)
print("VALID LEN", nb_valid_samples)

# x_train, y_train = train_generator()
# x_val, y_val = val_generator()
# train_datagen.fit(x_train)
# train_gen = train_datagen.flow(x_train,y_train, batch_size=batch_size)
# val_gen = train_datagen.flow(x_val,y_val, batch_size=batch_size)
train_gen = train_generator()
val_gen = val_generator()


with tf.device('/gpu:3'):
    # base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    # x = base_model.output
    # base_model.layers.pop()
    # x = base_model.layers[-1].output
    #
    # a = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=1)(x)
    # b = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2)(x)
    # c = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=4)(x)
    # d = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=6)(x)
    #
    # a_new = GlobalAveragePooling2D()(a)
    # b_new = GlobalAveragePooling2D()(b)
    # c_new = GlobalAveragePooling2D()(c)
    # d_new = GlobalAveragePooling2D()(d)
    #
    # merged = Add()([a_new, b_new, c_new, d_new])
    #
    # predictions = Dense(11, activation='sigmoid')(merged)
    #
    # model = Model(inputs=base_model.input, outputs=predictions)

    base_model = InceptionV3(weights='imagenet', include_top=False)#, input_shape=(image_size, image_size, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(11, activation=tf.nn.softmax)(x) #tf.nn.softmax
    model = Model(inputs=base_model.input, outputs=predictions)

#
# # first: train only the top layers (which were randomly initialized) i.e. freeze all convolution VGG16 layers
# for layer in base_model.layers:
#     layer.trainable = False

# # let's visualize layer names and layer indices to see how many layers we should freeze:
# for i, layer in enumerate(model.layers):
#    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:307]:
   layer.trainable = False
for layer in model.layers[307:]:
   layer.trainable = True

# model.compile(loss='mean_squared_error', optimizer='sgd')   SGD(lr=0.001, momentum=0.9)
model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['acc'])#SGD(lr=0.001)
# model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
model.fit_generator(train_gen, epochs=20, steps_per_epoch= nb_train_samples // batch_size+ 1,callbacks=[es_callback], validation_data=val_gen, validation_steps = nb_valid_samples // batch_size+ 1, verbose=1)
# model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=64, verbose=1) # callbacks=[tensorBoard,checkpoint],

model.save('food11-model-NODILATED-softmax.h5')


