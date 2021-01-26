import tensorflow as tf
from tensorflow.keras import layers

# ---- Generator model ----

# OLD VERSION
'''''
def make_generator_model():

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(8, (5, 5), strides=(1, 1), padding='same', input_shape=(512, 512, 1)))
    assert model.output_shape == (None, 512, 512, 8)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    def layer(depth, conv) :
        w = model.output_shape[1]
        model.add(layers.Conv2D(depth, (conv, conv), strides=(1, 1), padding='same'))
        assert model.output_shape == (None, w, w, depth)
        model.add(layers.LeakyReLU())

    layer(16, 5)
    layer(16, 5)
    layer(16, 5)
    layer(1, 5)

    return model
'''''

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
    conv2d_1 = layers.Conv2D(8, (5, 5), strides=(1, 1), padding='same', input_shape=(512, 512, 3))(inputs)
    batch_norm = layers.BatchNormalization()(conv2d_1)
    leaky_relu_1 = layers.LeakyReLU()(batch_norm)

    def layer(depth, conv, prev_layer):
        tmp = layers.Conv2D(depth, (conv, conv), strides=(1, 1), padding='same')(prev_layer)
        tmp = layers.BatchNormalization()(tmp)
        return layers.LeakyReLU()(tmp)

    tmp = layer(16, 5, leaky_relu_1)
    tmp = layer(16, 5, tmp)
    tmp = layer(16, 5, tmp)
    tmp = layer(16, 5, tmp)

    tmp = layers.Conv2D(1, (7, 7), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(tmp)

    model = tf.keras.Model(
                inputs=[masked_img_input, mask_input, noise_input],
                outputs=[tmp]
            )

    return model


# def make_generator_model():
#     model = tf.keras.Sequential()
#     model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(100,)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())
#
#     model.add(layers.Reshape((4, 4, 1024)))
#     assert model.output_shape == (None, 4, 4, 1024) # Note: None is the batch size
#
#     def layer(depth, conv, stride):
#         w = model.output_shape[1]
#         model.add(layers.Conv2DTranspose(depth, (conv, conv), strides=(stride, stride), padding='same', use_bias=False))
#         assert model.output_shape == (None, w * stride, w * stride, depth)
#         model.add(layers.BatchNormalization())
#         model.add(layers.LeakyReLU())
#
#     layer(1024, 7, 1)   # 4x4
#     layer(512, 7, 2)    # 8x8
#     layer(256, 7, 2)    # 16x16
#     layer(128, 7, 2)    # 32x32
#     layer(64, 7, 2)     # 64x64
#     layer(32, 7, 2)     # 128
#     layer(16, 7, 2)     # 256
#
#     model.add(layers.Conv2DTranspose(1, (7, 7), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
#     assert model.output_shape == (None, 512, 512, 1)
#
#     return model


# ---- Generator loss function ----
def generator_loss(fake_output, input, output, mask):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    context_loss = cross_entropy(input * mask, output * mask)
    percept_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    return context_loss + 0.1 * percept_loss
