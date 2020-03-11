
from matplotlib.pyplot import imread, imshow
import matplotlib.pyplot as plt

# Both have 600 x 800 x 3 shape
cimg = imread('data/louvre.jpg')
simg = imread('data/monet_800600.jpg')

# See image
#imshow(simg)
#plt.show()

# Assigning probabilities to specific hidden layers
# to affect our Generated image style
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

