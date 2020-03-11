# Supress TF CUDA errors
# see: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.applications import VGG19


def vgg19_model(input_tensor, input_shape):
    # Modify input_shape=(224,224,3)
    # Note: "width and height should be no smaller than 3"!
    # weights="imagenet" to use pretrained network
    vgg19 = VGG19(input_tensor=input_tensor, include_top=False,
                 input_shape=input_shape, weights="imagenet")
    
    return vgg19

# Testing
from tensorflow.keras.layers import Input
input_shape = (600, 800, 3)
test = Input(input_shape)
model = vgg19_model(test, input_shape)
import pdb
pdb.set_trace()

# See all layer outputs
# outputs = [layer.output for layer in model.layers]

# Get specific layer by name
# model.get_layer('block5_conv4')
# See layer names:
# model.summary()

# Take layers output (type = tf.Tensor):
# model.layers[1].output
