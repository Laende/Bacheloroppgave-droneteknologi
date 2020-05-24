from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, ReLU, Dense
from tensorflow.keras.initializers import TruncatedNormal


def make_generator(latent_dim=100):
    model = keras.Sequential()
    init = TruncatedNormal(mean=0.0, stddev=0.02)
    # Foundation for 26x26 images
    model.add(Dense(units=52*52*512,
                    use_bias=False,
                    kernel_initializer=init,
                    input_shape=(latent_dim,)))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU(alpha=0.2))

    model.add(keras.layers.Reshape(target_shape=(52, 52, 512)))

    model.add(Conv2DTranspose(filters=256,
                              kernel_size=(4, 4),
                              kernel_initializer=init,
                              strides=(1, 1),
                              padding='same',
                              use_bias=False))

    # model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(rate=0.4))

    # Upsample to 104x104
    model.add(Conv2DTranspose(filters=256,
                              kernel_size=(4, 4),
                              kernel_initializer=init,
                              strides=(2, 2),
                              padding='same',
                              use_bias=False))
    # model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(rate=0.4))

    model.add(Conv2DTranspose(filters=128,
                              kernel_size=(4, 4),
                              kernel_initializer=init,
                              strides=(2, 2),
                              padding='same',
                              use_bias=False))
    # model.add(BatchNormalization())
    model.add(ReLU())

    # Upsample to 416x416
    model.add(Conv2DTranspose(filters=128,
                              kernel_size=(4, 4),
                              kernel_initializer=init,
                              strides=(2, 2),
                              padding='same',
                              use_bias=False))
    # model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv2D(3,
                     (3, 3),
                     kernel_initializer=init,
                     activation='tanh',
                     padding='same'))
    assert model.output_shape == (None, 416, 416, 3)

    return model
