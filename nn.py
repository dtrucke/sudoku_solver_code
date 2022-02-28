import os
from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_HASH_VALUE
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


# smallest CNN
def create_model_small():
    checkpoint_path = "training_2/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  

    image_index = 7777 # You may select anything up to 60,000
    print(y_train[image_index]) # The label is 8
    print(x_test[19].size)
    plt.imshow(x_train[image_index], cmap='Greys')
    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
      
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    input_shape=(28,28,1)

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))

    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])




    return model

def LeNet_modified():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  

    image_index = 7777 # You may select anything up to 60,000
    print(y_train[image_index]) # The label is 8
    print(x_test[19].size)
    plt.imshow(x_train[image_index], cmap='Greys')
    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
      
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    input_shape=(28,28,1)

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])
# 784 - [32C3-32C3-32C5S2] - [64C3-64C3-64C5S2] - 128 - 10

    
    model= Sequential()

    model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])



def load_model(model):
    global trained_model
    trained_model = tf.keras.models.load_model(model)
    #scores = model.evaluate(X_test, y_test, verbose=0)

    return trained_model

def smaller_LeNet():
    model= Sequential()

    model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return


def train_model(debug=False):

    epochs=45

    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  

    image_index = 7777 # You may select anything up to 60,000
    print(y_train[image_index]) # The label is 8
    print(x_test[19].size)
    plt.imshow(x_train[image_index], cmap='Greys')
    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
      
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    input_shape=(28,28,1)

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=45,
        zoom_range = 0.1,  
        width_shift_range=0.1, 
        height_shift_range=0.1,
        fill_mode='nearest',
        validation_split = 0.2)

    datagen.fit(x_train)
    train_generator = datagen.flow(x_train, y_train, batch_size=60, subset='training')
    validation_generator = datagen.flow(x_test, y_test, batch_size=60, subset='validation')
    
    if debug==True:
        for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
            # create a grid of 3x3 images
            for i in range(0, 9):
                pyplot.subplot(330 + 1 + i)
                pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
            # show the plot
            pyplot.show()
            break
    
    
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x+epochs))

    model = LeNet_modified()
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                              save_weights_only=True,
    #                                              verbose=1)
    print("started")
    model.fit(train_generator,validation_data=validation_generator,
                    epochs = epochs,
                    callbacks=[annealer], verbose=0)

    model.evaluate(x_test, y_test)

   
def test_fkt(test_img):
    model_fin=load_model()
    pred=0 

    start_find=time.perf_counter()
    pred = model_fin.predict(test_img.reshape(1, 28, 28, 1))
    #print(pred)
    test=np.array(pred[0])
    max=pred.argmax()
    if np.max(test)>0.97:
        cell=pred.argmax()
        #print(pred.argmax())
    else:
        print("Zelle leer")
        cell=0

    end_find=time.perf_counter()
    print(end_find-start_find)
    return cell

def test():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  

    image_index = 7777 # You may select anything up to 60,000
    print(y_train[image_index]) # The label is 8
    print(x_test[19].size)
    #plt.imshow(x_train[image_index], cmap='Greys')
    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
      
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    input_shape=(28,28,1)

    # # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

    model = load_model("LeNet5_modified.h5")
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print("LeNet5_modified.h5, accuracy: {:5.2f}%".format(100 * acc))
    model = load_model("small_CNN.h5")
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print("small_CNN.h5, accuracy: {:5.2f}%".format(100 * acc))
    model = load_model("LeNet5_modified_smaller.h5")
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print("LeNet5_modified_smaller.h5, accuracy: {:5.2f}%".format(100 * acc))


def load_model(model):
    global trained_model
    trained_model = tf.keras.models.load_model(model)
    #scores = model.evaluate(X_test, y_test, verbose=0)
    




    return trained_model



#test()