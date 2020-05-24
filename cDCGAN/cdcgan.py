import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from numpy.random import randn, randint
from decimal import Decimal
from tqdm import tqdm
import pickle
from tensorflow.keras.utils import plot_model
from cDCGAN.Discriminator import make_discriminator
from cDCGAN.Generator import make_generator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

IM_SIZE = 32*13
OUTPUT_DIR = "./img"
IMAGE_DIR = 'IMAGE DIR'

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)

    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)

    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)

    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)

    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))

    # save plot
    save_plot(x_fake, epoch)
    # save the generator model
    filename = f'./models/generator_model_{epoch+1}.h5'
    g_model.save(filename)


# create and save a plot of generated images
def save_plot(examples, epoch, n=1):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i])
    # save plot to file
    filename = f'./plots/generated_plot_epoch_{epoch+1}.png'
    plt.savefig(filename)
    plt.close()


# example of smoothing class=1 to [0.8, 1.2]
def smooth_positive_labels(y):
    return y - 0.3 + (np.random.random(y.shape) * 0.3)


def smooth_negative_labels(y):
    return y + np.random.random(y.shape) * 0.3


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# load and prepare cifar10 training images
def load_real_samples():
    with open('train_images.pkl', 'rb') as f:
        train_images = pickle.load(f)

    train_images = train_images.reshape(train_images.shape[0], IM_SIZE, IM_SIZE, 3).astype('float32')

    # train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_images


# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    y = smooth_positive_labels(y)
    return X, y


def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)

    # predict outputs
    X = generator.predict(x_input)

    # create 'fake' class labels (0)
    y = np.zeros((n_samples, 1))
    y = smooth_negative_labels(y)
    return X, y


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False

    # connect them
    model = keras.Sequential()

    # add generator
    model.add(g_model)

    # add the discriminator
    model.add(d_model)

    # compile model
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=56):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in tqdm(range(n_epochs), leave=True):
        # enumerate batches over the training set
        for j in range(bat_per_epo):

            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)

            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)

            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)

            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))

            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print(f'Epoch: {i+1}, '
                  f'batch pr. epoch: {j+1}/{bat_per_epo} '
                  f'd1= {round(Decimal(str(d_loss1)), 5)}, '
                  f'd2= {round(Decimal(str(d_loss2)), 5)} '
                  f'g= {round(Decimal(str(g_loss)), 5)}')

        if (i + 1) % 40 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # Size of latent space
    latent_dim = 100
    # Create the generator model
    generator = make_generator(latent_dim=latent_dim)

    # Create the discriminator model
    diskriminator = make_discriminator()

    generator.summary()
    diskriminator.summary()
    plot_model(generator, to_file='generator-model-plot.png', show_shapes=True, show_layer_names=True)
    plot_model(diskriminator, to_file='diskriminator-model-plot.png', show_shapes=True, show_layer_names=True)

    # Create the GAN
    gan_model = define_gan(g_model=generator, d_model=diskriminator)
    # plot_model(diskriminator, to_file='diskriminator-model-plot.png', show_shapes=True, show_layer_names=True)

    # Load images of pipes with ice
    samples = load_real_samples()

    # train model
    train(generator, diskriminator, gan_model, samples, latent_dim, n_epochs=40000, n_batch=4)
