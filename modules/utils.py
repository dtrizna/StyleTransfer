import tensorflow as tf

def gram_matrix(A):
    """
    A -- unrolled matrix of shape (n_C, n_H * n_W)
    GA -- Gram matrix of shape (n_C, n_C)
    """
    return tf.matmul(A, tf.transpose(A))
