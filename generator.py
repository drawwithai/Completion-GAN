from layers import *

import tensorflow as tf
from tensorflow.keras import layers
import math
import numpy as np

"""
Definition of Generator's Model.

It take three inputs of ( masked_image, mask, random_noise ) each of shape 256x256x1
And output an image with the same shape

The model consist of two facing funnels, with a kind of blender in the midle

The first part consist of cascading convolutions layers to compress image to a more abstract form.
The second part is a pipe of dilated convolution layers, the idea was to propagate global image features to prepare the reconstruction.
The last part is a cascade of deconvolution layers, converting abstract features to a new image.

Noise input should be moved further inside the model instead of with the image and the mask.

Loss function is a bit weird.
It's a linear combinaison of three losses :

  - preceipt_loss : measure efficiency at fooling the discriminator
  - context_loss : try to measure if the non masked area is preserved
  - fillrate : try to punish the generator if he doesn't try to draw in the masked area
"""

def make_generator_model():
    """
    : return : the generator as a tf.keras.Model
    """

    # SETTINGS :
    IMGRES = 256  # Input images resolution
    depth = 32    # 

    # ---- Defining inputs ----
    # Masked image
    masked_img_input = tf.keras.Input(shape=(IMGRES, IMGRES, 1), name="Maskedimage")
    # Mask image
    mask_input = tf.keras.Input(shape=(IMGRES, IMGRES, 1), name="Maskimage")
    # Random seed
    noise_input = tf.keras.Input(shape=(IMGRES, IMGRES, 1), name="Noise")

    # ---- Merge all inputs in one ----
    inputs = layers.Concatenate(axis=3)([masked_img_input, mask_input, noise_input])

    # ---- Defining layers ----
    tmp = layers.Conv2D(
                3,
                (5, 5),
                strides=(1, 1),
                padding='same',
                input_shape=(IMGRES, IMGRES, 3)
            )(inputs)
    tmp = layers.BatchNormalization()(tmp)
    tmp = layers.LeakyReLU()(tmp)


    # 256
    tmp = Convolution(tmp, depth * 1, 3)  # 128
    tmp = Convolution(tmp, depth * 2, 3)  # 64
    tmp = Convolution(tmp, depth * 4, 3)  # 32

    tmp = DilatedConvolution(tmp, depth * 8, 5) # 32
    tmp = DilatedConvolution(tmp, depth * 8, 5) # 32
    tmp = DilatedConvolution(tmp, depth * 8, 5) # 32
    tmp = DilatedConvolution(tmp, depth * 8, 5) # 32

    tmp = Deconvolution(tmp, depth * 4, 3)  # 64
    tmp = Deconvolution(tmp, depth * 2, 3)  # 128
    tmp = Deconvolution(tmp, depth * 1, 3)  # 256

    tmp = layers.Conv2D(
                1,
                (7, 7),
                strides=(1, 1),
                padding='same',
                use_bias=False,
                activation='tanh'
            )(tmp)

    model = tf.keras.Model(
                inputs=[masked_img_input, mask_input, noise_input],
                outputs=[tmp]
            )

    return model


# ---- Generator loss function ----
def generator_loss(fake_output, input, output, mask):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    qty_img = tf.reduce_sum(input * mask)  # Sum of pixels in masked area
    qty_mask = tf.reduce_sum(mask)
    fillrate = qty_img / qty_mask

    k = 4
    a = 2
    fillrate = k * (fillrate - 0.5) ** a

    context_loss = cross_entropy(input * mask, output * mask)
    percept_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    return context_loss + 0.1 * percept_loss + fillrate
