#!/usr/bin/python3.7

import tensorflow as tf
print(tf.__version__)

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

import tensorflow_datasets as tfds

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.config.threading.set_inter_op_parallelism_threads(16)

#db =

#train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
#train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 1024)))
    assert model.output_shape == (None, 4, 4, 1024) # Note: None is the batch size


    def layer(depth, conv, stride) :
        w = model.output_shape[1]
        model.add(layers.Conv2DTranspose(depth, (conv, conv), strides=(stride, stride), padding='same', use_bias=False))
        assert model.output_shape == (None, w * stride, w * stride, depth)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

    layer(1024, 5, 1)   # 4x4
    layer(512, 5, 2)    # 8x8
    layer(256, 5, 2)    # 16x16
    layer(128, 5, 2)    # 32x32
    layer(64, 5, 2)     # 64x64
    layer(32, 5, 2)     # 128
    layer(16, 5, 2)     # 256

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 512, 512, 1)

    return model

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

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


generator = make_generator_model()
discriminator = make_discriminator_model()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
discriminator_optimizer=discriminator_optimizer,
generator=generator,
discriminator=discriminator)

# Training loop

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

BUFFER_SIZE = 60000
BATCH_SIZE = 1

# Batch and shuffle the data
train_dataset = tfds.load('tensorflowdb', split='train', as_supervised=True, batch_size=BATCH_SIZE, shuffle_files=True, download=False)

for i in train_dataset:
  print(i)
  break

def normalize_image(image, label):
  return (tf.cast(image, tf.float32) - 127.5) / 127.5, label
train_dataset = train_dataset.map(normalize_image)

for i in train_dataset:
  print(i)
  break

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        print("Start epoch :", epoch)

        for image_batch in dataset:
            train_step(image_batch[0])

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
            epoch + 1,
            seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        generate_and_save_images(generator,
            epochs,
            seed)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()

train(train_dataset, EPOCHS)
