#!/usr/bin/python3.7
import generator as gan_generator
from generator import *
import discriminator as gan_discriminator
import masks
from masks import *
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
import argparse
import random

parser = argparse.ArgumentParser(description='Mask all images from input dir to output dir')
parser.add_argument('batch', type=int, help='Batch Size')
parser.add_argument('datadir', type=str, nargs='?')

args = parser.parse_args()

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
BATCH_SIZE = args.batch
num_examples_to_generate = BATCH_SIZE

# ---- Loading Dataset ----
def normalize_image(img):
    if isinstance(img, dict):  # elements from dataset are dict
        return (tf.cast(img.get('image'), tf.float32) - 127.5) / 127.5
    else:  # mask (or other directly loaded images) are not dict
        return (tf.cast(img, tf.float32) - 127.5) / 127.5

def load_image45() :

    ds = tfds.load('oneline45', split='train', as_supervised=False, batch_size=None, shuffle_files=True, download=False)
    ds = ds.map(normalize_image, num_parallel_calls=-1)

    return ds

def load_images(path) :

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=1)
        # resize the image to the desired size
        return tf.image.resize(img, [512, 512])

    def process_path(file_path):
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        img = (tf.cast(img, tf.float32) - 127.5) / 127.5
        return img

    ds = tf.data.Dataset.list_files(path, shuffle=False)
    ds = ds.map(process_path, num_parallel_calls=-1)
    ds = ds.shuffle(BUFFER_SIZE)

    return ds

# def generateMaskedDataset(images):

    # N = tf.data.experimental.cardinality(images)
    # NBMASK = 16

    # masks = tf.data.Dataset.from_tensor_slices([generate_random_mask() for _ in range(NBMASK)])
    # masks = masks.repeat(N // NBMASK + int((N % NBMASK) != 0))

    # noise = tf.data.Dataset.from_tensors(tf.random.normal([512, 512, 1]))
    # noise = noise.repeat(N)

    # maskeds = images

    # def apply_mask(img, mask, noise) :
        # return (img * mask, mask, noise)

    # ds = tf.data.Dataset.zip((maskeds, masks, noise)).map(apply_mask, num_parallel_calls=-1)

    # return ds

def batch_and_fetch_dataset(ds) :

    return ds.batch(BATCH_SIZE).cache().prefetch(-1)

if args.datadir is not None :
    images = load_images(args.datadir + '*')
    print(" >>>>>> Load custom dataset :", args.datadir)
else :
    images = load_image45()
    print(" >>>>>> Load default dataset : oneline45")

dataset = batch_and_fetch_dataset(images)

print(" >> dataset : ", dataset)

# ---- Get random mask ----

# ---- Tensorboard ----
logdir = 'logs'  # folder where to put logs
writer = tf.summary.create_file_writer(logdir)

generator_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
discriminator_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)

# ---- generates set of masks ----
maskarray = [generate_random_mask() for _ in range(BATCH_SIZE)]
maskarray = tf.stack(maskarray)

@tf.function
def train_step(images, masks):
    print(" IMAGES :    ", images)

    def process_image(img, masks):
        # Randomly transform image
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        rand_nb = random.randint(0, 3)
        for i in range(rand_nb):
            img = tf.image.rot90(img)

        # Generate and apply mask
        img_masked = img * masks

        # Generate random noise
        noise = tf.random.normal([BATCH_SIZE, 512, 512, 1])

        return [img, (img_masked, maskarray, noise)]


    # Process all images and put them in 2 tables
    # images = images.map(process_image)
    images = process_image(images, masks)
    print(" >> images processing done << : ", images)
    full = images[0]
    masked = images[1]

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(masked, training=True)

        real_output = discriminator(full, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = gan_generator.generator_loss(fake_output, masked[0], generated_images, masked[1])
        disc_loss = gan_discriminator.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    generator_metric(gen_loss)
    discriminator_metric(disc_loss)

    return images[1]


def train(fullimages, masks, epochs):

    # Restore checkpoint if found
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(" >>>>> Restored from {}".format(manager.latest_checkpoint))
    else:
        print(" >>>>> Initializing from scratch.")

    fullitr = iter(fullimages)

    for epoch in range(epochs):
        start = time.time()
        print(" >> >  > Starting epoch : ", epoch, " << <  < ")
        try:
            fullbatch = next(fullitr)
        except StopIteration:
            fullitr = iter(fullimages)
            fullbatch = next(fullitr)

        maskbatch = train_step(fullbatch, masks)

        template = '--> Epoch {}, Generator Loss: {}, Discriminator Loss: {}'
        print (template.format(epoch+1,
            generator_metric.result(),
            discriminator_metric.result()))

        generator_metric.reset_states()
        discriminator_metric.reset_states()

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 maskbatch)

        # Save the model every 15 epochs
        checkpoint.step.assign_add(1)
        if (epoch + 1) % 25 == 0:
            path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), path))

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        # generate_and_save_images(generator,
                                 # 9999,
                                 # maskbatch)

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
    plt.close()
    #plt.show()

if not os.path.exists('./results'):
    os.mkdir('./results')

train(dataset, maskarray, EPOCHS)
