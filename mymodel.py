# Supress TF CUDA errors
# see: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from matplotlib.pyplot import imread, imshow
import matplotlib.pyplot as plt
from modules.transfer import vgg19_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from modules.utils import generate_noise_image, reshape_and_normalize_image
from modules.cost import compute_content_cost, compute_style_cost_one_layer, total_cost
import pdb

# ==============
# LOADING | GENERATING IMAGES
# Both have 600 x 800 x 3 shape
cimg = reshape_and_normalize_image(imread('data/louvre.jpg'))
simg = reshape_and_normalize_image(imread('data/monet_800600.jpg'))
gimg = generate_noise_image(cimg)


#print(cimg.shape, simg.shape, gimg.shape)

# See image
#imshow(gimg[0])
#plt.show()


# Assigning probabilities to specific hidden layers
# to affect our Generated image style
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]


vgg19 = vgg19_model(cimg[0].shape)
# Holds model: tensorflow.python.keras.engine.training.Model 

# ==========================
# COMPUTE CONTENT COST

# Set specific layer in NN we're interested in (point taken from Assignment)
out = vgg19.get_layer('block4_conv2').output
# Value: Tensor("block4_conv2/Identity:0", shape=(None, 75, 100, 512), dtype=float32)

content_model = Model(inputs=vgg19.input, outputs=out)
# Holds model: tensorflow.python.keras.engine.training.Model
# with same structure as vgg19, but gives output from specific layer!

a_C = content_model.predict(cimg)
# Value: actual activations in form of numpy.ndarray
# These are already evaluated activations

a_G = out
# Value: Tensor("block4_conv2/Identity:0", shape=(None, 75, 100, 512), dtype=float32)
# Theis is Tensor (handle) in same NN part as a_C
# will be filled layer

J_content = compute_content_cost(a_C, a_G)
# Value: Tensor("truediv_1:0", shape=(), dtype=float32)


# ==========================
# COMPUTE STYLE COST

J_style = 0

for layer_name, weight in STYLE_LAYERS:
    
    out = vgg19.get_layer(layer_name).output
    style_model = Model(inputs=vgg19.input, outputs=out)

    a_S = style_model.predict(simg)
    # actual activations: array
    
    a_G = out
    # Tensor representing same position within NN

    J_style_layer = compute_style_cost_one_layer(a_S, a_G)
    J_style += weight * J_style_layer


# =================
# Total cost
J = total_cost(J_content, J_style)

# =================

opt = Adam(lr=2, beta_1=0.9, beta_2=0.999, decay=0.01)

# HOW TO FEED MODEL OUR OWN LOSS FUNCTIONWITH THIS OPTIMIZER

# TRAIN NETWORK ON OUR GENERATE IMAGE AS INPUT AND UPDATE IT (HOW?)

pdb.set_trace()


# here you feed your data?
#vgg19.fit_generator()
