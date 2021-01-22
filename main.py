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

tf.keras.utils.plot_model(generator, "generator.png", show_shapes=True)
tf.keras.utils.plot_model(discriminator, "discriminator.png", show_shapes=True)

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
EPOCHS = 200
BUFFER_SIZE = 60000
BATCH_SIZE = 32
num_examples_to_generate = BATCH_SIZE

# Batch and shuffle the data
# masked_dataset = tfds.load('tensorflowdb', split='train', as_supervised=True, batch_size=BATCH_SIZE, shuffle_files=True, download=False)


def load_online45() :
    ds = tfds.load('oneline45', split='train', as_supervised=False, batch_size=BATCH_SIZE, shuffle_files=True, download=False)

    def normalize_image(ele):
        return (tf.cast(ele.get('image'), tf.float32) - 127.5) / 127.5, ele.get('label')

    ds = ds.map(normalize_image)

    for i in ds:
        print(">>>>> masked_dataset normalized : ", i)
        break

    return ds

def load_folder(path, masks_path=None) :

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=1)
        # resize the image to the desired size
        return tf.image.resize(img, [512, 512])

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # Integer encode the label
        return file_path

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        img = (tf.cast(img, tf.float32) - 127.5) / 127.5
        return img

    if masks_path is None :

        ds = tf.data.Dataset.list_files(path, shuffle=False)
        ds = ds.map(process_path, num_parallel_calls=-1)
        ds = ds.shuffle(BUFFER_SIZE)

    else :

        ds = tf.data.Dataset.list_files(path, shuffle=False)
        ds = ds.map(process_path, num_parallel_calls=-1)
        ds = ds.shuffle(BUFFER_SIZE)

        mask = tf.data.Dataset.list_files(masks_path, shuffle=False)
        mask = mask.map(process_path, num_parallel_calls=-1)
        mask = mask.repeat(ds.cardinality()).shuffle(BUFFER_SIZE)

        ds = tf.data.Dataset.zip((ds, mask))

    ds = ds.batch(BATCH_SIZE).cache().prefetch(-1)

    print(ds)
    # for i in ds:
    #     print(">>>>> dataset normalized : ", i)
    #     break

    return ds

masked_dataset = load_folder('data/masked/*', 'data/masks/*')
full_dataset = load_folder('data/full/*')

seed = next(iter(masked_dataset))

# ---- Tensorboard ----
# logdir = 'logs'  # folder where to put logs
# writer = tf.summary.create_file_writer(logdir)

# def normalize_image(img, label):
#     return (tf.cast(img, tf.float32) - 127.5) / 127.5, label

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

    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    maskitr = iter(maskedimages)
    fullitr = iter(fullimages)

    for epoch in range(epochs):
        start = time.time()

        print(" >>>>> Starting epoch : ", epoch)

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

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 9999,
                                 seed)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input[0], training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        if 16 <= i : break
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('./results/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
    #plt.show()

if not os.path.exists('./results'):
    os.mkdir('./results')
train(masked_dataset, full_dataset, EPOCHS)
