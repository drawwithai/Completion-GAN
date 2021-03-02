import tensorflow as tf
import tensorflow_datasets as tfds

# ---- Normalize Images ----
def normalize_image(img):
    if isinstance(img, dict):  # elements from dataset are dict
        return (tf.cast(img.get('image'), tf.float32) - 127.5) / 127.5
    else:  # mask (or other directly loaded images) are not dict
        return (tf.cast(img, tf.float32) - 127.5) / 127.5


# ---- Load oneline45 dataset (deprecated) ----
def load_image45() :
    ds = tfds.load('oneline45', split='train', as_supervised=False, batch_size=None, shuffle_files=True, download=False)
    ds = ds.map(normalize_image, num_parallel_calls=-1)

    return ds


# ---- Load dataset from local images ----
def load_images(path, buffer_size):

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
    ds = ds.shuffle(buffer_size)

    return ds


def batch_and_fetch_dataset(ds, batch_size):
    return ds.batch(batch_size).cache().prefetch(-1)
