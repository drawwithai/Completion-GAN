import tensorflow as tf
import tensorflow_datasets as tfds

"""
Contains utilities usefull to manipulate and create the training dataset
"""

# ---- Normalize Images ----
def normalize_image(img):
    """
    Transform image's pixels from [0;255] to [-1;1]
    : return : Tensor
    """
    if isinstance(img, dict):  # elements from dataset are dict
        return (tf.cast(img.get('image'), tf.float32) - 127.5) / 127.5
    else:  # mask (or other directly loaded images) are not dict
        return (tf.cast(img, tf.float32) - 127.5) / 127.5


# ---- Load oneline45 dataset (deprecated) ----
def load_image45() :
    """
    Load a small test dataset -- DEPRECATED
    """
    ds = tfds.load('oneline45', split='train', as_supervised=False, batch_size=None, shuffle_files=True, download=False)
    ds = ds.map(normalize_image, num_parallel_calls=-1)

    return ds


# ---- Load dataset from local images ----
def load_images(path, buffer_size):
    """
    Load all images in given folder and return a tansorflow dataset 
    all images must have a resolution of 256x256
    : return : tf.data.Dataset
    """

    def process_path(file_path):
        """
        Read a single image and return it as normalised tf.Tensor
        """
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, [256, 256])
        img = (tf.cast(img, tf.float32) - 127.5) / 127.5
        return img

    ds = tf.data.Dataset.list_files(path, shuffle=False)
    ds = ds.map(process_path, num_parallel_calls=-1)
    ds = ds.shuffle(buffer_size).repeat()

    return ds


def batch_and_fetch_dataset(ds, batch_size):
  """
  Batch a dataset according to given batch size and cache it content
  : return : tf.data.CachedDataset
  """
  return ds.batch(batch_size).cache().prefetch(-1)
