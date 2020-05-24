import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from acgan.Discriminator import discriminator
from acgan.Generator import generator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

BATCH_SIZE = 256
EPOCHS = 600
NUM_EXAMPLES_TO_GENERATE = 4
NOISE_DIM = 100
LEARNING_RATE_GENERATOR = 0.0002
LEARNING_RATE_DISCRIMINATOR = 0.0002
BETA_1 = 0.5
HEIGHT = WIDTH = 64
CHANNELS = 3
N_CLASSES = 2
DATASET_PATH = 'DATASET PATH'
VALIDATION_PATH = 'VALIDATION PATH'


def _parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize(image_decoded, (HEIGHT, WIDTH))
    image_normalized = (image_resized - 127.5) / 127.5
    return image_normalized, label


def ac_gan(latent_size=100):
    optimizer = Adam(lr=0.0002, beta_1=0.5)

    discriminator_model = discriminator(input_shape=(HEIGHT, WIDTH, CHANNELS), n_classes=N_CLASSES)
    discriminator_model.compile(
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
        optimizer=optimizer)

    discriminator_model.summary()
    plot_model(discriminator_model, to_file='./acgan/discriminator_plot.png', show_shapes=True, show_layer_names=True)

    generator_model = generator(latent_size)
    generator_model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer)

    generator_model.summary()
    plot_model(generator_model, to_file='./acgan/generator_plot.png', show_shapes=True, show_layer_names=True)

    latent = Input(shape=(latent_size,), name='latent_noise')
    image_class = Input(shape=(1,), dtype='int32', name='image_class')

    fake_img = generator_model([latent, image_class])

    discriminator_model.trainable = False

    fake, aux = discriminator_model(fake_img)

    combined = Model(inputs=[latent, image_class],
                     outputs=[fake, aux],
                     name='ACGAN')

    combined.compile(
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
        optimizer=optimizer)
    combined.summary()
    plot_model(combined, to_file='./acgan/acgan_plot.png', show_shapes=True, show_layer_names=True)

    return combined, discriminator_model, generator_model


def print_logs(metrics_names, train_history, test_history):
    print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
        'component', *metrics_names))
    print('-' * 65)

    ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
    print(ROW_FMT.format('generator (train)',
                         *train_history['generator'][-1]))
    print(ROW_FMT.format('generator (test)',
                         *test_history['generator'][-1]))
    print(ROW_FMT.format('discriminator (train)',
                         *train_history['discriminator'][-1]))
    print(ROW_FMT.format('discriminator (test)',
                         *test_history['discriminator'][-1]))


def get_data(path):
    imagepaths, labels = list(), list()
    classes = sorted(os.walk(path).__next__()[1])
    class_map = {name: idx for idx, name in enumerate(classes)}
    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(path, c)
        walk = os.walk(c_dir).__next__()
        # Add each image to the training set
        for sample in walk[2]:
            # Only keeps jpeg images
            if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                imagepaths.append(os.path.join(c_dir, sample))
                labels.append(class_map[c])

    filenames = tf.constant(imagepaths)
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    return dataset


if __name__ == '__main__':
    validation_dataset = get_data(VALIDATION_PATH)
    validation_dataset_length = [i for i, _ in enumerate(validation_dataset)][-1] + 1
    validation_dataset = validation_dataset.shuffle(validation_dataset_length).batch(BATCH_SIZE)
    print(f"Dataset for validation consists of {validation_dataset_length} images.")

    combined, discriminator_model, generator_model = ac_gan(latent_size=NOISE_DIM)

    generator_model.load_weights('WEIGHTS PATH')
    discriminator_model.load_weights('WEIGHTS PATH')

    noise = np.tile(np.random.uniform(-1, 1, (2, NOISE_DIM)), (2, 1))
    sampled_labels = np.array([[i] * 2 for i in range(2)]).reshape(-1, 1)

    generated_images = generator_model.predict([noise, sampled_labels], verbose=0)

    for counter, image in enumerate(generated_images):
        plt.subplot(2, 2, 1 + counter)
        plt.axis('off')
        plt.imshow(image)
    plt.show()
