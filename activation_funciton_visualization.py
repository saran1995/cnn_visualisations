
from __future__ import print_function
import tensorflow as tf
import  matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.applications.vgg16 import preprocess_input

import time
import numpy as np
from PIL import Image as pil_image
from keras.preprocessing.image import save_img
from keras import layers
from keras.applications import vgg16
from keras import backend as K
from keras import Model
from keras.models import load_model
import matplotlib.pyplot as plt







def Activation_Visualizations(image , model, layer_name):
    def draw_activations(filters, predict_image):
        ix = 1
        square = int(np.floor(np.sqrt(filters)))
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(predict_image[0, :, :, ix - 1], cmap='gray')
                ix += 1
        # show the figure
        return plt



    # def predict(self, image, model, layer_name):
    #     # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    output_layer = layer_dict[layer_name]
    model = Model(inputs=model.inputs, outputs=output_layer.output)
    model.summary()
    filter = model.get_weights()[1].shape
    class_id = model.predict(image)
    saved_img = draw_activations(filter, class_id)

    return saved_img








