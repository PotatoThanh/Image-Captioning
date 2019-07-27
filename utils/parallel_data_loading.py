import tensorflow as tf
import numpy as np
from utils.data_processing import load_image
# Load the numpy files
def map_func(img_name, cap):
#   img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    img_tensor, _ = load_image(img_name)
    return img_tensor, cap

def parallel_loading(img_name_train, cap_train, BUFFER_SIZE, BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # using map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.py_func(
            map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=4)

    # shuffling and batching
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    return dataset