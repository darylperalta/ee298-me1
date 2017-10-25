import tensorflow as tf
import pickle
import keras
import math
import glob
import os
from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, ELU, BatchNormalization, Lambda, merge, MaxPooling2D, Input, Activation
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
import keras.backend as K
import numpy as np

#batch norm, relu, update optimizer to SGD
def homography_regression_model():
    input_shape=(128, 128, 2)
    input_img = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), name='conv1',padding ='same', activation='relu')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), name='conv2', padding = 'same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(64, (3, 3), name='conv3', padding ='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), name='conv4', padding='same',activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)

    x = Conv2D(128, (3, 3), name='conv5', padding='same',activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), name='conv6',padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(x)

    x = Conv2D(128, (3, 3), name='conv7', padding ='same',activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), name='conv8', padding = 'same',activation='relu')(x)

    x = Dropout(0.5)(x)
    x = Flatten()(x)

    x = Dense(1024, name='FC1', activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(8, name='loss')(x)
    #out = Dropout(0.5)(x)
    model = Model(inputs=input_img, outputs = out)
    #plot_model(model, to_file='HomegraphyNet_Regression.png', show_shapes=True)
    model.compile(optimizer=keras.optimizers.SGD(lr=0.005, momentum=0.9), loss='MSE')

    #model.compile(optimizer=Adam(lr=1e-3), loss=euclidean_distance,metrics =['accuracy'])
    return model

model = homography_regression_model()
model.summary()

batch_size = 64
epochs = 10

def data_loader(path, batch_size=64, normalize=True):
    """Generator to be used with model.fit_generator()"""
    while True:
        files = glob.glob(os.path.join(path, '*.npz'))
        np.random.shuffle(files)
        for npz in files:
            # Load pack into memory
            archive = np.load(npz)
            images = archive['images']
            offsets = archive['offsets']
            del archive
            #_shuffle_in_unison(images, offsets)
            # Split into mini batches
            num_batches = int(len(offsets) / batch_size)
            images = np.array_split(images, num_batches)
            offsets = np.array_split(offsets, num_batches)
            while offsets:
                batch_images = images.pop()
                batch_offsets = offsets.pop()
                if normalize:
                    batch_images = (batch_images - 127.5) / 127.5
                    batch_offsets = batch_offsets / 32.
                yield batch_images, batch_offsets


# Dataset-specific
train_data_path = '/media/airscan/Data/AIRSCAN/darwin_data/192.168.1.103/~darwin/train-set'
#test_data_path = '/media/airscan/Data/AIRSCAN/darwin_data'
num_samples = 65 * 7680 # 158 archives x 3,072 samples per archive, but use just 150 and save the 8 for testing
#7680
# From the paper
batch_size = 64
#total_iterations = 90000

steps_per_epoch = num_samples / batch_size # As stated in Keras docs
#steps_per_epoch = 3
#epochs = int(total_iterations / steps_per_epoch)
epochs = 12


filepath="/media/airscan/Data/AIRSCAN/weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1)
#model.fit_generator(gen(), steps_per_epoch = 1562, epochs = 1, verbose=1, callbacks=[LearningRateScheduler(step_decay)], validation_data=None, class_weight=None, nb_worker=1)
# Train
model.fit_generator(data_loader(train_data_path, batch_size),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs, verbose =1, callbacks=[checkpoint])

# serialize model to JSON
model_json = model.to_json()
with open("/media/airscan/Data/AIRSCAN/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/media/airscan/Data/AIRSCAN/model.h5")
print("Saved model to disk")

#score = model.evaluate(X_valid, y_valid, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
print("Finished Training")
