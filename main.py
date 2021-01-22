#!/usr/bin/python3.7
import generator as gan_generator
from generator import *
import discriminator as gan_discriminator
from discriminator import *
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
import PIL
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import callbacks
import time
from IPython import display
import tensorflow_datasets as tfds
import cv2

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
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

# ---- Training loops settings ----
EPOCHS = 500
BUFFER_SIZE = 60000
BATCH_SIZE = 5
num_examples_to_generate = BATCH_SIZE

# ---- Loading Dataset ----
dataset = tfds.load('oneline45', split='train', as_supervised=False, batch_size=BATCH_SIZE, shuffle_files=True, download=False)

# ---- MASKING ----
# ---- Preparing mask ----
# Generating mask
white_bg = np.ones([512, 512], dtype=np.uint8)
white_bg.fill(255)
mask = cv2.circle(white_bg, (260, 300), 225, (0,0,0), -1)
mask = cv2.bitwise_and(white_bg, mask)
mask = tf.Variable(mask, dtype='uint8')
mask = tf.reshape(mask, [512, 512, 1])  # We need mask to have only 1 channel, not 3

# ---- Normalizing mask and every image of dataset ----
def normalize_image(ele):
    if isinstance(ele, dict):  # elements from dataset are dict
        return (tf.cast(ele.get('image'), tf.float32) - 127.5) / 127.5
    else:  # mask (or other directly loaded images) are not dict
        return (tf.cast(ele, tf.float32) - 127.5) / 127.5
    

mask = normalize_image(mask)
mask = tf.add(tf.multiply(mask, 0.5), 0.5)  # set max to 0 - 1 values instead of 0 - 255
print(" >>>>> Normalized mask")
dataset = dataset.map(normalize_image)
print(" >>>>> Normalized dataset")

# ---- Creating masked dataset ----
dataset_masked = dataset
dataset_masked = dataset_masked.map(lambda ele: ele * mask)  # multiply image by mask to apply it
print(" >>>>> Masked images of dataset")

# ---- Generating seed ----
seed = next(iter(dataset_masked))
print(">>>>> Seed : ", seed)

# ---- Tensorboard ----
# logdir = 'logs'  # folder where to put logs
# writer = tf.summary.create_file_writer(logdir)

generator_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
discriminator_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)

@tf.function
def train_step(masked, full):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(masked, training=True)

        real_output = discriminator(full, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = gan_generator.generator_loss(fake_output)
        disc_loss = gan_discriminator.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    generator_metric(gen_loss)
    discriminator_metric(disc_loss)

def train(maskedimages, fullimages, epochs):

    # Restore checkpoint if found
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(" >>>>> Restored from {}".format(manager.latest_checkpoint))
    else:
        print(" >>>>> Initializing from scratch.")

    maskitr = iter(maskedimages)
    fullitr = iter(fullimages)

    for epoch in range(epochs):
        start = time.time()
        print(" > > > > > Starting epoch : ", epoch)
        try:
            maskbatch = next(maskitr)
            fullbatch = next(fullitr)
        except StopIteration:
            maskitr = iter(maskedimages)
            fullitr = iter(fullimages)
            maskbatch = next(maskitr)
            fullbatch = next(fullitr)

        train_step(maskbatch, fullbatch)

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
        checkpoint.step.assign_add(1)
        if (epoch + 1) % 25 == 0:
            path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), path))

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 9999,
                                 seed)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        if 16 <= i : break
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('./results/image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()

if not os.path.exists('./results'):
    os.mkdir('./results')

train(dataset_masked, dataset, EPOCHS)
