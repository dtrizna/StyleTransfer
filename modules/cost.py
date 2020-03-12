import tensorflow as tf
from modules.utils import gram_matrix

def compute_content_cost(a_C, a_G):
    """
    a_C - tensor of dimension (1,n_H,n_W,n_C), 
        representing X hidden layer activation of content image (C)
    a_G - tensor of dimension (1,n_H,n_W,n_C), 
        representing X hidden layer activation of generated image (C) 
    """
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.reshape(a_C, shape=[1, -1, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[1, -1, n_C])

    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))/(4 * n_H * n_W * n_C)

    return J_content

def compute_style_cost_one_layer(a_S, a_G):
    """
    a_S - tensor of dimension (1,n_H,n_W,n_C), 
        representing X hidden layer activation of style image (C)
    a_G - tensor of dimension (1,n_H,n_W,n_C), 
        representing X hidden layer activation of generated image (C) 
    """
    J_style_layer = 0

    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]), perm=[1, 0])
    a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]), perm=[1, 0])

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG)))/(4 * n_C**2 * (n_W * n_H)**2)
    
    return J_style_layer
   

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    return alpha * J_content + beta * J_style


if __name__ == "__main__":
    # Check:
    tf.random.set_seed(1)
    print(compute_content_cost(tf.random.normal([1,2,3,4]),tf.random.normal([1,2,3,4])))