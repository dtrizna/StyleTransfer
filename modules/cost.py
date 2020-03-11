import tensorflow as tf
from utils import gram_matrix

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


def compute_style_cost(model, STYLE_LAYERS):
    """
    STYLE_LAYERS -- list with tuples as elements, where each tuple's:
                    [0] -- contains layer name
                    [1] -- weight to assign that layer in order to compute cost
                Note! Overall sum of all weights should be 1.
    """
    J_style = 0
    for name, weight in STYLE_LAYERS:
        a_S_layer = model.get_layer(name).output
        


# Check:
tf.random.set_seed(1)
print(compute_content_cost(tf.random.normal([1,2,3,4]),tf.random.normal([1,2,3,4])))
print(compute_style_cost(tf.random.normal([1,2,3,4]),tf.random.normal([1,2,3,4])))
