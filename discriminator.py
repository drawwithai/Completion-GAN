import tensorflow as tf
from tensorflow.keras import layers

# ---- Discriminator model ----
def make_discriminator_model():

    # SETTINGS :
    IMGRES = 256

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(8, (5, 5), strides=(2, 2), padding='same', input_shape=(IMGRES, IMGRES, 1)))
    assert model.output_shape == (None, IMGRES / 2, IMGRES / 2, 8)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    def layer(depth, conv, stride) :
        w = model.output_shape[1]
        model.add(layers.Conv2D(depth, (conv, conv), strides=(stride, stride), padding='same'))
        assert model.output_shape == (None, w / stride, w / stride, depth)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

    x = 4
    
    layer(16 * x, 5, 2)
    layer(32 * x, 5, 2)
    layer(64 * x, 5, 2)
    layer(128 * x, 5, 2)
    layer(256 * x, 5, 2)

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# ---- Discriminator loss function ----
def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss, real_loss, fake_loss
