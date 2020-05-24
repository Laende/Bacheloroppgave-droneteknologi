from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Input, Dropout
from tensorflow.keras.layers import Dense, Flatten, multiply, Embedding, Reshape
from tensorflow.keras.models import Model

def generator(latent_size, n_classes=2):
    # Weight init
    init = TruncatedNormal(mean=0.0, stddev=0.02)

    def up_sampling_block(x, filter_size, kernel_size=(5, 5), strides=(2, 2), activation='relu', batch_norm=True, dropout=False):
        x = Conv2DTranspose(filters=filter_size,
                            kernel_size=kernel_size,
                            strides=strides,
                            kernel_initializer=init,
                            padding='same',
                            activation=activation)(x)
        if batch_norm:
            x = BatchNormalization()(x)

        if dropout:
            x = Dropout(0.5)(x)
        return x

    # Input 1
    # image class label
    image_class = Input(shape=(1, ), dtype='int32', name='image_class')

    # class embeddings
    emb = Embedding(n_classes,
                    latent_size,
                    embeddings_initializer='glorot_normal')(image_class)

    # 10 classes in MNIST
    cls = Flatten()(emb)

    # Input 2
    # latent noise vector
    latent_input = Input(shape=(latent_size, ), name='latent_noise')

    # hadamard product between latent embedding and a class conditional embedding
    h = multiply([latent_input, cls])

    # Conv generator
    x = Dense(1024, activation='relu')(h)
    x = Dense(256 * 8 * 8, activation='relu')(x)
    x = Reshape((8, 8, 256))(x)

    # upsample to (16, 16, 128)
    x = up_sampling_block(x, filter_size=256)

    # upsample to (32, 32, 256)
    x = up_sampling_block(x, filter_size=128)

    # upsample to (64, 64, 256)
    x = up_sampling_block(x, filter_size=64)

    # reduce channel into binary image (28, 28, 1)
    generated_img = Conv2DTranspose(filters=3,
                                    kernel_size=(2, 2),
                                    strides=(1, 1),
                                    padding='same',
                                    activation='tanh')(x)
    model = Model(inputs=[latent_input, image_class],
                  outputs=generated_img,
                  name='generator')
    return model
