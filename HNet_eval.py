import tensorflow as tf
import pickle
import keras
import math
import glob
import os
from keras.models import Sequential, Model, load_model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, ELU, BatchNormalization, Lambda, merge, MaxPooling2D, Input, Activation
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
import keras.backend as K
import numpy as np


test_data_path = '/media/airscan/Data/AIRSCAN/darwin_data/192.168.1.103/~darwin/test-set'

model = load_model('/media/airscan/Data/AIRSCAN/weights-improvement-11.hdf5')

def mean_ave_corner_error(y_true, y_pred):
    return K.mean(32*K.sqrt(K.sum(K.square(K.reshape(y_pred,[-1,4,2]) - K.reshape(y_true,[-1,4,2])), axis=1, keepdims=True)))

def data_loader(path, batch_size=64, normalize=True):
    while True:
        files = glob.glob(os.path.join(path, '*.npz'))
        np.random.shuffle(files)
        for npz in files:
            # Load pack into memory
            archive = np.load(npz)
            images = archive['images']
            offsets = archive['offsets']
            del archive
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

model.compile(optimizer=keras.optimizers.SGD(lr=0.005, momentum=0.9), loss='MSE', metrics = [mean_ave_corner_error])
model.summary()
score = model.evaluate_generator(data_loader(test_data_path, 64),
                         steps=400)
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
print("%s: %0.2f" % (model.metrics_names[0], score[0]))
print("%s: %0.2f" % (model.metrics_names[1], score[1]))
