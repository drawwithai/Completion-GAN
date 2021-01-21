#!/usr/bin/python3.7

import generator as gan_generator
from generator import *
import discriminator as gan_discriminator
from discriminator import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import callbacks
import time
from IPython import display
import tensorflow_datasets as tfds

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.config.threading.set_inter_op_parallelism_threads(16)

# ---- Creating optimizers ----
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# ---- Creating generator and discriminator ----
generator = gan_generator.make_generator_model()
discriminator = gan_discriminator.make_discriminator_model()

# ---- Checkpoints settings ----
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# ---- Training loops settings ----
EPOCHS = 500
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim]) # random seed, we use the same over time

BUFFER_SIZE = 60000
BATCH_SIZE = 16

# Batch and shuffle the data
# train_dataset = tfds.load('tensorflowdb', split='train', as_supervised=True, batch_size=BATCH_SIZE, shuffle_files=True, download=False)

def load_online45() :
    ds = tfds.load('oneline45', split='train', as_supervised=False, batch_size=BATCH_SIZE, shuffle_files=True, download=False)

    def normalize_image(ele):
        return (tf.cast(ele.get('image'), tf.float32) - 127.5) / 127.5, ele.get('label')

    ds = ds.map(normalize_image)

    for i in ds:
        print(">>>>> train_dataset normalized : ", i)
        break

    return ds

train_dataset = load_online45()

# ---- Tensorboard ----
# logdir = 'logs'  # folder where to put logs
# writer = tf.summary.create_file_writer(logdir)

# def normalize_image(img, label):
#     return (tf.cast(img, tf.float32) - 127.5) / 127.5, label

generator_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
discriminator_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = gan_generator.generator_loss(fake_output)
        disc_loss = gan_discriminator.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    generator_metric(gen_loss)
    discriminator_metric(disc_loss)

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        print(" >>>>> Starting epoch : ", epoch)

        for image_batch in dataset:
            # ---------- Tensorboard graph
            # tf.summary.trace_on(graph=True, profiler=True)
            # ---------- Tensorboard graph
            train_step(image_batch[0])
            # ---------- Tensorboard graph
            # with writer.as_default():
                # tf.summary.trace_export(
                        # name="Training my model !",
                        # step=0,
            #             profiler_outdir='logs/')
            # ---------- Tensorboard graph

            # ---- Tensorboard ----
            # with writer.as_default():
                # tf.summary.flush(writer=None)

        template = 'Epoch {}, Generator Loss: {}, Discriminator Loss: {}'
        print (template.format(epoch+1,
            generator_metric.result(),
            discriminator_metric.result()))

        generator_metric.reset_states()
        discriminator_metric.reset_states()

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 50 == 0:
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

    plt.savefig('results/image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()

os.mkdir('results')
train(train_dataset, EPOCHS)
