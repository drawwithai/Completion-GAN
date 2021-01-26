import random
import cv2
import tensorflow as tf
import numpy as np


def generate_random_mask():

    HEIGHT = 512
    WIDTH = 512
    MARGIN = 100  # margin on border on canvas

    # ---- Creating white background ----
    white_bg = np.ones([WIDTH, HEIGHT], dtype=np.uint8)
    white_bg.fill(255)

    def random_circle_mask():
        radius = np.random.randint(50, 150)
        posX = np.random.randint(radius + MARGIN, WIDTH - radius)
        posY = np.random.randint(radius + MARGIN, HEIGHT - radius)
        mask = cv2.circle(white_bg, (posX, posY), radius, (0, 0, 0), -1)
        return mask

    def random_rectangle_mask():
        start = (np.random.randint(MARGIN, WIDTH - MARGIN),
                 np.random.randint(MARGIN, HEIGHT - MARGIN))
        endX = np.random.randint(
                    0 + 2 * MARGIN,
                    WIDTH - 2 * MARGIN
                )
        endY = np.random.randint(
                    0 + 2 * MARGIN,
                    WIDTH - 2 * MARGIN
                )
        if endX >= start[0]:
            endX += MARGIN
        else:
            endX -= MARGIN
        if endY >= start[1]:
            endY += MARGIN
        else:
            endY -= MARGIN

        mask = cv2.rectangle(white_bg, start, (endX, endY), (0, 0, 0), -1)

        return mask

    # ---- Pick a random mask function in list ----
    mask_list = [
            random_circle_mask,
            random_rectangle_mask]
    mask = random.choice(mask_list)()

    mask = tf.Variable(mask, dtype='uint8')
    mask = tf.reshape(mask, [512, 512, 1])  # We need mask to have only 1 channel, not 3
    mask = tf.cast(mask, tf.float32) / 255

    return mask
