import tensorflow as tf
import numpy as np

def gram_matrix(A):
    """
    A -- unrolled matrix of shape (n_C, n_H * n_W)
    GA -- Gram matrix of shape (n_C, n_C)
    """
    return tf.matmul(A, tf.transpose(A))

class CONFIG:
    IMAGE_WIDTH = 800
    IMAGE_HEIGHT = 600
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.8
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 

def generate_noise_image(content_image, noise_ratio = CONFIG.NOISE_RATIO):
    noise_image = np.random.uniform(-100, 100, (1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)).astype('float32')
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image

def reshape_and_normalize_image(image):
    image = np.reshape(image, ((1,) + image.shape))
    image = image - CONFIG.MEANS
    return image