import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
from collections import defaultdict
from acgan.Discriminator import discriminator
from acgan.Generator import generator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input
import numpy as np
import tensorflow as tf
import time


BATCH_SIZE = 512
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
    train_dataset = get_data(DATASET_PATH)
    train_dataset_length = [i for i, _ in enumerate(train_dataset)][-1] + 1
    train_dataset = train_dataset.shuffle(train_dataset_length).batch(BATCH_SIZE)
    print(f"Dataset for training consists of {train_dataset_length} images.")

    validation_dataset = get_data(VALIDATION_PATH)
    validation_dataset_length = [i for i, _ in enumerate(validation_dataset)][-1] + 1
    validation_dataset = validation_dataset.shuffle(validation_dataset_length).batch(BATCH_SIZE)
    print(f"Dataset for validation consists of {validation_dataset_length} images.")

    combined, discriminator_model, generator_model = ac_gan(latent_size=NOISE_DIM)

    train_history = defaultdict(list)
    test_history = defaultdict(list)
    if not os.path.exists('/logs'):
        os.makedirs(os.path.join('/logs'))
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        epoch_gen_loss = []
        epoch_disc_loss = []
        epoch_disc_val_loss = []
        epoch_gen_val_loss = []
        i = 0
        for training_image_batch in train_dataset:
            start_time = time.time()
            training_images, training_labels = training_image_batch
            noise = tf.random.normal([len(training_images), NOISE_DIM])
            sampled_labels = np.random.randint(0, N_CLASSES, len(training_images))

            generated_images = generator_model.predict([noise, sampled_labels.reshape((-1, 1))])

            X = np.concatenate((training_images, generated_images))
            y = np.array([1] * len(training_images) + [0] * len(training_images))
            aux_y = np.concatenate((training_labels, sampled_labels), axis=0)

            # train discriminator
            d_loss = discriminator_model.train_on_batch(X, [y, aux_y])
            epoch_disc_loss.append(d_loss)
            ### Train Generator ###
            # generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = tf.random.normal([len(training_images) * 2, NOISE_DIM])
            sampled_labels = np.random.randint(0, N_CLASSES, len(training_images) * 2)

            # we want to train the generator to trick the discriminator
            # so all the labels should be not-fake (1)
            trick = np.ones(2 * len(training_images))
            gan_loss = combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))],
                [trick, sampled_labels])

            epoch_gen_loss.append(gan_loss)
            i += 1
            print(f"Step: {i}/{int(np.ceil(train_dataset_length / BATCH_SIZE))}, time pr. step: {round(time.time() - start_time, 2)}s, estimated time pr epoch: {round(((time.time() - start_time) * int(np.ceil(train_dataset_length / BATCH_SIZE)))/60, 2)} min")

        print('\nTesting for epoch {}:'.format(epoch + 1))
        for validation_image_batch in validation_dataset:
            # Evaluate Discriminator
            validation_images, validation_labels = validation_image_batch
            # generate a new batch of noise
            noise = tf.random.normal([len(validation_images), NOISE_DIM])
            sampled_labels = np.random.randint(0, N_CLASSES, len(validation_images))

            generated_images = generator_model.predict(
                [noise, sampled_labels.reshape((-1, 1))],
                verbose=False)

            # construct discriminator evaluation dataset
            X = np.concatenate((validation_images, generated_images))
            y = np.array([1] * len(validation_images) + [0] * len(validation_images))
            aux_y = np.concatenate((validation_labels, sampled_labels), axis=0)

            # evaluate discriminator
            # test loss
            discriminator_test_loss = discriminator_model.evaluate(X, [y, aux_y], verbose=False)
            epoch_disc_val_loss.append(discriminator_test_loss)
            # -------------------------------------------------------------------------------------------
            # Evaluate Generator
            # make new noise
            noise = tf.random.normal([len(validation_images) * 2, NOISE_DIM])
            sampled_labels = np.random.randint(0, N_CLASSES, len(validation_images) * 2)

            # create labels
            trick = np.ones(2 * len(validation_images))

            # evaluate generator
            generator_test_loss = combined.evaluate(
                [noise, sampled_labels.reshape((-1, 1))],
                [trick, sampled_labels], verbose=False)
            epoch_gen_val_loss.append(generator_test_loss)

        # train loss
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        # Test loss
        discriminator_test_loss = np.mean(np.array(epoch_disc_val_loss), axis=0)
        generator_test_loss = np.mean(np.array(epoch_gen_val_loss), axis=0)

        # append train losses
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        # append test losses
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        # print training and test losses
        print_logs(discriminator_model.metrics_names, train_history, test_history)

        # save weights every epoch
        generator_model.save_weights(filepath='WEIGHTS PATH'.format(epoch), overwrite=True)
        discriminator_model.save_weights(filepath='WEIGHTS PATH'.format(epoch), overwrite=True)

        # Save train test loss history
        pickle.dump({'train': train_history, 'test': test_history}, open('TRAIN HISTORY PATH', 'wb'))
