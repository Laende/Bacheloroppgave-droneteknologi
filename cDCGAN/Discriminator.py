from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dropout, LeakyReLU, Flatten, Dense
from tensorflow.keras.initializers import TruncatedNormal


def make_discriminator(input_shape=(416, 416, 3), lr=0.0002, beta_1=0.5):
    model = keras.Sequential()
    init = TruncatedNormal(mean=0.0, stddev=0.02)

    model.add(Conv2D(filters=48,
                     kernel_size=(4, 4),
                     kernel_initializer=init,
                     strides=(1, 1),
                     padding='same',
                     input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))

    # Downsample
    model.add(Conv2D(filters=128,
                     kernel_size=(4, 4),
                     kernel_initializer=init,
                     strides=(2, 2),
                     padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # downsample
    model.add(Conv2D(filters=256,
                     kernel_size=(4, 4),
                     kernel_initializer=init,
                     strides=(2, 2),
                     padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(rate=0.4))

    model.add(Conv2D(filters=256,
                     kernel_size=(4, 4),
                     kernel_initializer=init,
                     strides=(2, 2),
                     padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(rate=0.4))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    optimizer = keras.optimizers.Adam(lr=lr, beta_1=beta_1)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
