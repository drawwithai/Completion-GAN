import tensorflow as tf
from tensorflow.keras import layers

# ---- Discriminator model ----
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(8, (5, 5), strides=(2, 2), padding='same', input_shape=(512, 512, 1)))
    assert model.output_shape == (None, 256, 256, 8)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    def layer(depth, conv, stride) :
        w = model.output_shape[1]
        model.add(layers.Conv2D(depth, (conv, conv), strides=(stride, stride), padding='same'))
        assert model.output_shape == (None, w / stride, w / stride, depth)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

    layer(16, 5, 2) # 128x128
    layer(32, 5, 2) # 64x64
    layer(64, 5, 2) # 32x32
    layer(128, 5, 2) # 16x16
    layer(256, 5, 2) # 8x8
    layer(512, 5, 2) # 4x4

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# ---- Discriminator loss function ----
def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss
