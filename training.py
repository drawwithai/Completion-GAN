import time
import random
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt
from masks import generate_random_mask


# ---- Creating optimizers ----
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(generator, discriminator, images, batch_size):
    
    # ---- Creating metrics for monitoring ----
    generator_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
    discriminator_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)

    # print(" IMAGES :    ", images)  # DEBUG

    # ---- generates set of masks ----
    masks = [generate_random_mask() for _ in range(batch_size)]
    masks = tf.stack(masks)

    # --- Process images before training ---
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
        noise = tf.random.normal([batch_size, 512, 512, 1])

        return [img, (img_masked, masks, noise)]


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

        gen_loss = generator.generator_loss(fake_output, masked[0], generated_images, masked[1])
        disc_loss = discriminator.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    generator_metric(gen_loss)
    discriminator_metric(disc_loss)

    return images[1]


# ---- Global training of models ----
def train(
        generator,
        discriminator,
        fullimages,
        epochs,
        batch_size,
        checkpoint,
        checkpoint_manager
        ):
    
    # ---- Metrics ----
    generator_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
    discriminator_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)

    # Restore checkpoint if found
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print(" >>>>> Restored from {}".format(checkpoint_manager.latest_checkpoint))
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

        maskbatch = train_step(generator, discriminator, fullbatch, batch_size)

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
            path = checkpoint_manager.save()
            print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), path))

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        # generate_and_save_images(generator,
                                 # 9999,
                                 # maskbatch)


# TODO : enregistrer que des images solo def
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

