import tensorflow as tf
from tensorflow.keras import layers

"""
Definition of the Discriminator's model

Actualy, a simple convolution funnel followed by dense layers
It takes a 256x256x1 normalised image as input 
and output a single value, <0 if predicted fake, >0 if real

The loss function is binary crossentropy, summ of loss on real and fake images
"""

# ---- Discriminator model ----
def make_discriminator_model():
    """
    : return : the discriminator model as a tf.keras.Model
    """

    # SETTINGS :
    IMGRES = 256    # Input tensors size
    size_mult = 2   # Factor to increase or decrease convolution layer's depths
    kernel = 5      # Basic kernel size for convolution layers

    # The input layer has an arbitrary depth of 8
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(8 * size_mult, (kernel, kernel), strides=(2, 2), padding='same', input_shape=(IMGRES, IMGRES, 1)))
    assert model.output_shape == (None, IMGRES / 2, IMGRES / 2, 8 * size_mult)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3)) # dropout arbitrary set to 0.3

    def layer(depth, conv, stride) :
        w = model.output_shape[1]
        model.add(layers.Conv2D(depth, (conv, conv), strides=(stride, stride), padding='same'))
        assert model.output_shape == (None, w / stride, w / stride, depth)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

    layer(16 * size_mult, kernel, 2)
    layer(32 * size_mult, kernel, 2)
    layer(64 * size_mult, kernel, 2)
    layer(128 * size_mult, kernel, 2)
    layer(256 * size_mult, kernel, 2)

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# ---- Discriminator loss function ----
def discriminator_loss(real_output, fake_output):
    """
    Discriminator's loss function using binary crossentropy
    : return : tuple of (total_loss, loss_on_real_images, loss_on_fake_images)
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss, real_loss, fake_loss
