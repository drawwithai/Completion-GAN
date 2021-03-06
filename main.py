#!/usr/bin/python3.7
from masks import *
from generator import *
from discriminator import *
from dataset_utils import *
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
import os

"""
Completion GAN's main file.

USAGE :

  main batch_size [images_folder]

    batch_size : 
        Number of images to take at once for training. 
        Must be tweaked according to available ram memory.
        Higher is better.
        
    images_folder :
        If given, it will train on images in given folder.
        Load the dataset oneline45 if not given.

Divided in three parts : 

  setup : Create models and load the dataset
  train : Training and monitoring functions
  main  : Launch the training

Traning settings are available below :

  EPOCHS                    : Number of train steps to perform
  BUFFER_SIZE               : smth used to cache dataset, defaut value on Tensorflow tutorials
  BATCH_SIZE                : Number of images to take at once. Must be tweaked according to available ram memory
  num_examples_to_generate  : Number of images generated to monitor generator outputs
  IMGRES                    : Images resolution. Don't change this unless you're ready to tweak all values (and probably debug things)

"""

# Disable GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ---- Manage launch arguments ----
parser = argparse.ArgumentParser(description='Mask all images from input dir to output dir')
parser.add_argument('batch', type=int, help='Batch Size')
parser.add_argument('datadir', type=str, nargs='?')
args = parser.parse_args()

# ---- Training loops settings ----
EPOCHS = 500
BUFFER_SIZE = 60000
BATCH_SIZE = args.batch
num_examples_to_generate = BATCH_SIZE
IMGRES = 256

# ---- Initialization ----
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.threading.set_inter_op_parallelism_threads(16)

# ---- Creating optimizers ----
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# ---- Creating generator and discriminator ----
generator = make_generator_model()
discriminator = make_discriminator_model()

# Save graphs of models as png
tf.keras.utils.plot_model(generator, "generator.png", show_shapes=True)
tf.keras.utils.plot_model(discriminator, "discriminator.png", show_shapes=True)

# ---- Checkpoints settings ----
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
        step=tf.Variable(1),
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

# ---- Loading Dataset ----
# Loads local images if specified,
# otherwise loads oneline45 dataset (deprecated)
if args.datadir is not None:
    images = load_images(args.datadir + '*', BUFFER_SIZE)
    print(" >>>>>> Load custom dataset :", args.datadir)
else :
    images = load_image45()
    print(" >>>>>> Load default dataset : oneline45")

dataset = batch_and_fetch_dataset(images, BATCH_SIZE)

# print(" >> dataset : ", dataset)

# ---- Tensorboard ----
logdir = 'logs'  # folder where to put logs
writer = tf.summary.create_file_writer(logdir)

generator_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
discriminator_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)

# TODO: move it to train_step()
# ---- generates set of masks ----
maskarray = [generate_random_mask() for _ in range(BATCH_SIZE)]
maskarray = tf.stack(maskarray)


@tf.function
def train_step(images, masks):

    # --- Images processing ----
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
        noise = tf.random.normal([BATCH_SIZE, 256, 256, 1])

        return img, (img_masked, masks, noise)

    # Process all images and put them in 2 tables
    full, masked = process_image(images, masks)
    # print(" >> images processing done << : ", images)

    # ---- Gradient descent ----
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(masked, training=True)

        real_output = discriminator(full, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output, masked[0], generated_images, masked[1])
        disc_loss, real_loss, fake_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    generator_metric(gen_loss)
    discriminator_metric(disc_loss)

    return masked, fake_output, real_output, real_loss, fake_loss


# ---- TRAIN THE MODELS ----
def train(fullimages, masks, epochs):

    # ---- Checkpoint restoration ----
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(" >>>>> Restored from {}".format(manager.latest_checkpoint))
    else:
        print(" >>>>> Initializing from scratch.")

    fullitr = iter(fullimages)

    for epoch in range(epochs):
        start = time.time()
        try:
            fullbatch = next(fullitr)
        except StopIteration:
            fullitr = iter(fullimages)
            fullbatch = next(fullitr)

        # ---- Actual training ----
        maskbatch, fake_output, real_output, real_loss, fake_loss = train_step(fullbatch, masks)

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

        fake_acuracy = 0
        threshold = 5
        for result in fake_output:
            fake_acuracy += int(result < -threshold)

        real_acuracy = 0
        for result in real_output:
            real_acuracy += int(threshold < result)

        print()
        print("======= Epoch : %4d =======" % (epoch +1))

        print("Generator Loss: %3.3f, Discriminator Loss: %3.3f" % (generator_metric.result(),discriminator_metric.result()))
        print("Took %3.3f sec" % (time.time()-start))

        col = 4
        n = (4 + 3*col + 6*col +1)
        print("=" * n)
        print("from | %-6s | %-7s | %-7s | %-7s" % ("scores", "min", "max", "avg"))
        print("fake | %3.0f%%   | %+7.3f | %+7.3f | %+7.3f" % (fake_acuracy * 100 / BATCH_SIZE, np.min(fake_output), np.max(fake_output), np.average(fake_output)))
        print("real | %3.0f%%   | %+7.3f | %+7.3f | %+7.3f" % (real_acuracy * 100 / BATCH_SIZE, np.min(real_output), np.max(real_output), np.average(real_output)))
        print("=" * n)

        generator_metric.reset_states()
        discriminator_metric.reset_states()

        # ---- Generate after the final epoch ----
        display.clear_output(wait=True)


# ---- Generate images and save them as png ----
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(2,2), dpi=300)

    for i in range(predictions.shape[0]):
        if 4 <= i : break
        plt.subplot(2, 2, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('./results/image_at_epoch_{:04d}.png'.format(epoch))
    plt.savefig('./results/00_last.png')
    plt.close()
    #plt.show()

if __name__ == "__main__" :

  # Make sure that the results folder exists
  if not os.path.exists('./results'):
      os.mkdir('./results')

  # Call the training
  train(dataset, maskarray, EPOCHS)
