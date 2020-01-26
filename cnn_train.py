# -------------------------- set gpu using tf ---------------------------
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)




import keras.backend.tensorflow_backend as K
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator, save_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
import Visualizations
from numpy import expand_dims
import matplotlib.pyplot as plt
from skimage.io import imsave
from activation_funciton_visualization import Activation_Visualizations

batch_size = 512
num_classes = 10
epochs = 1000
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
conv_layer_name = ['conv2d_2', 'conv2d_4', 'conv2d_6']
test_image = x_train[3]
test_image = img = expand_dims(test_image, axis=0)

def save_as_image(img_flat, fname, PIXELS_DIR):
    """
        Saves a data blob as an image file.
    """

    # consecutive 1024 entries store color channels of 32x32 image
    plt.figure(figsize=(30, 5))
    plt.imshow(img_flat.reshape(32,32,3))
    plt.savefig(os.path.join(fname, "test_img.png"), interpolation='nearest')

class Visualization(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        model = self.model
        print(epoch)
        if epoch % 10 == 0 and epoch > 0:

            for i in range(len(model.layers)):
                if i == 0:
                    # Skipping the input layer
                    continue
                history_cnn = tf.keras.callbacks.History()
                layer = model.layers[i]
                # check for convolutional layer
                if 'conv' not in layer.name or layer.name not in conv_layer_name:
                    continue

                # summarize output shape
                print(i, layer.name, layer.output.shape)
                save_dir = layer.name
                visualize_filter = Visualizations.visualize_layer(model, layer_name=layer.name)
                activation = Activation_Visualizations(test_image, model,  layer.name)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

                save_img(os.path.join(save_dir, "{0:}_{1:}.png".format(layer.name, epoch)), visualize_filter)
                # save_img(os.path.join(save_dir, "{0:}_activation_map_{1:}.png".format(layer.name, epoch)), activation)
                activation.savefig(os.path.join(save_dir, "{0:}_activation_map_{1:}.png".format(layer.name, epoch)))
                save_as_image(test_image, save_dir, 'test_image.png' )






def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))


    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

if  __name__ =="__main__":
    model = create_model()

    print(model.summary())
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        history_cnn = tf.keras.callbacks.History()
        checkpoint_cnn = tf.keras.callbacks.ModelCheckpoint(filepath="saved_models/keras_cifar10_trained_model.h5", save_best_only=True,
                                                            monitor="val_loss", save_weights_only=False,
                                                            mode="min")
        vis = Visualization()
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            workers=100, use_multiprocessing=False,
                            callbacks=[checkpoint_cnn,
                                      vis]

        )
