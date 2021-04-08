import tensorflow as tf
from tensorflow.keras import layers
import math
import numpy as np

# ---- Generator model ----

def make_generator_model():

    # SETTINGS :
    IMGRES = 256

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
    conv2d_1 = layers.Conv2D(
                3,
                (5, 5),
                strides=(1, 1),
                padding='same',
                input_shape=(IMGRES, IMGRES, 3)
            )(inputs)
    batch_norm = layers.BatchNormalization()(conv2d_1)
    leaky_relu_1 = layers.LeakyReLU()(batch_norm)

    def uplayer(prev_layer):
        tmp = layers.UpSampling2D(size=(2,2))(prev_layer)
        tmp = layers.BatchNormalization()(tmp)
        return layers.LeakyReLU()(tmp)

    def downlayer(prev_layer):
        tmp = layers.AveragePooling2D(pool_size=(2,2))(prev_layer)
        tmp = layers.BatchNormalization()(tmp)
        return layers.LeakyReLU()(tmp)

    tmp = downlayer(leaky_relu_1)
    tmp = downlayer(tmp)

    tmp = downlayer(tmp)
    tmp = downlayer(tmp)
    tmp = downlayer(tmp)

    tmp = uplayer(tmp)
    tmp = uplayer(tmp)
    tmp = uplayer(tmp)
    tmp = uplayer(tmp)
    tmp = uplayer(tmp)

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
