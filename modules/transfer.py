from tensorflow.keras.applications import VGG19

def vgg19_model(input_shape):
    # Modify input_shape=(224,224,3)
    # Note: "width and height should be no smaller than 3"!
    # weights="imagenet" to use pretrained network
    vgg19 = VGG19(include_top=False, input_shape=input_shape, weights="imagenet")
    
    return vgg19

# See all layer outputs
# outputs = [layer.output for layer in model.layers]

# Get specific layer by name
# model.get_layer('block5_conv4')
# See layer names:
# model.summary()

# Take layers output (type = tf.Tensor):
# model.layers[1].output
