import tensorflow as tf
from tensorflow.keras import layers

# ---- Generator model ----

def make_generator_model():

    # ---- Defining inputs ----
    # Masked image
    masked_img_input = tf.keras.Input(shape=(512, 512, 1), name="Maskedimage")
    # Mask image
    mask_input = tf.keras.Input(shape=(512, 512, 1), name="Maskimage")
    # Random seed
    noise_input = tf.keras.Input(shape=(512, 512, 1), name="Noise")

    # ---- Merge all inputs in one ----
    inputs = layers.Concatenate(axis=3)([masked_img_input, mask_input, noise_input])

    # ---- Defining layers ----
    conv2d_1 = layers.Conv2D(3, (5, 5), strides=(1, 1), padding='same', input_shape=(512, 512, 3))(inputs)
    batch_norm = layers.BatchNormalization()(conv2d_1)
    leaky_relu_1 = layers.LeakyReLU()(batch_norm)

    def uplayer(depth, conv, fac, prev_layer):
        tmp = layers.Conv2DTranspose(depth, (conv, conv), strides=(fac, fac), padding='same')(prev_layer)
        tmp = layers.BatchNormalization()(tmp)
        return layers.LeakyReLU()(tmp)

    def downlayer(depth, conv, fac, prev_layer):
        tmp = layers.Conv2D(depth, (conv, conv), strides=(fac, fac), padding='same')(prev_layer)
        tmp = layers.BatchNormalization()(tmp)
        return layers.LeakyReLU()(tmp)

    tmp = downlayer(64, 3, 2, leaky_relu_1)
    tmp = downlayer(128, 3, 2, tmp)
    tmp = downlayer(256, 3, 2, tmp)

    tmp = downlayer(256, 3, 1, tmp)
    tmp = downlayer(256, 3, 1, tmp)
    tmp = downlayer(256, 3, 1, tmp)

    tmp = uplayer(128, 3, 2, tmp)
    tmp = uplayer(64, 3, 2, tmp)
    tmp = uplayer(32, 3, 2, tmp)
    tmp = uplayer(8, 5, 1, tmp)

    tmp = layers.Conv2D(1, (7, 7), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(tmp)

    model = tf.keras.Model(
                inputs=[masked_img_input, mask_input, noise_input],
                outputs=[tmp]
            )

    return model


# ---- Generator loss function ----
def generator_loss(fake_output, input, output, mask):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    context_loss = cross_entropy(input * mask, output * mask)
    percept_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    return context_loss + 0.1 * percept_loss
