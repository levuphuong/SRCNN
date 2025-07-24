from keras.models import Sequential
from keras.layers import Conv2D, Activation, Input
from keras import backend as K

class SRCNN:
    @staticmethod
    def build(width, height, depth):
        # determine the input shape depending on the image data format
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        else:
            inputShape = (height, width, depth)

        # build the SRCNN model using the Sequential API
        # Layer 1: Conv2D with 64 filters of size 9x9 and ReLU activation
        # Layer 2: Conv2D with 32 filters of size 1x1 and ReLU activation
        # Layer 3: Conv2D output layer with depth equal to the number of image channels
        model = Sequential([
            Input(shape=inputShape),
            Conv2D(64, (9, 9), padding="valid", activation="relu", kernel_initializer='normal'),
            Conv2D(32, (1, 1), padding="valid", activation="relu", kernel_initializer='normal'),
            Conv2D(depth, (5, 5), padding="valid", kernel_initializer='normal')
        ])

        return model
