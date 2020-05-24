import math
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.layers import Conv2DTranspose, Dropout, Reshape, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from cDCGAN.utils.SpectralNormalizationKeras import SpectralNormalization

CLASS_MAP = {0: 'Airplane',
             1: 'Automobile',
             2: 'Bird',
             3: 'Cat',
             4: 'Deer',
             5: 'Dog',
             6: 'Frog',
             7: 'Horse',
             8: 'Ship',
             9: 'Truck'}
BATCH_SIZE = 8
EPOCHS = 1000
NUM_EXAMPLES_TO_GENERATE = 4
NOISE_DIM = 100
LEARNING_RATE_GENERATOR = 0.0003
LEARNING_RATE_DISCRIMINATOR = 0.0002
BETA_1 = 0.5


def generator_loss(fake_output, apply_label_smoothing=True):
    cross_entropy = BinaryCrossentropy(from_logits=False)
    if apply_label_smoothing:
        fake_output_smooth = smooth_negative_labels(tf.ones_like(fake_output))
        return cross_entropy(tf.ones_like(fake_output_smooth), fake_output)
    else:
        return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output, apply_label_smoothing=True, label_noise=True, loss_func='ralsgan'):
    real_output_modified = None
    fake_output_modified = None
    cross_entropy = BinaryCrossentropy(from_logits=False)
    if label_noise and not apply_label_smoothing:
        # Noisy labels:
        real_output_modified = noisy_labels(tf.ones_like(real_output), 0.06)
        fake_output_modified = noisy_labels(tf.zeros_like(fake_output), 0.06)

    elif apply_label_smoothing and not label_noise:
        # Smooth labels
        real_output_modified = smooth_positive_labels(real_output)
        fake_output_modified = smooth_negative_labels(fake_output)

    if label_noise and apply_label_smoothing:
        # Noisy labels:
        real_output_noise = noisy_labels(tf.ones_like(real_output), 0.06)
        fake_output_noise = noisy_labels(tf.zeros_like(fake_output), 0.06)

        # Smooth labels
        real_output_modified = smooth_positive_labels(real_output_noise)
        fake_output_modified = smooth_negative_labels(fake_output_noise)

    if loss_func == 'gan':
        real_loss = cross_entropy(tf.ones_like(real_output_modified), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output_modified), fake_output)

        loss = fake_loss + real_loss
        return loss
    elif loss_func == 'ralsgan':
        loss = (tf.reduce_mean(
            tf.square(real_output_modified - tf.reduce_mean(fake_output_modified) - tf.ones_like(real_output_modified)))
         + tf.reduce_mean(
                    tf.square(fake_output_modified - tf.reduce_mean(real_output_modified) + tf.ones_like(
                        fake_output_modified)))) / 2.
        return loss


def discriminator(spectral_normalization=True):
    model = tf.keras.Sequential()
    init = TruncatedNormal(mean=0.0, stddev=0.02)

    if spectral_normalization:
        model.add(SpectralNormalization(Conv2D(filters=128,
                                               kernel_size=(4, 4),
                                               kernel_initializer=init,
                                               strides=(2, 2),
                                               padding='same')))
        model.add(LeakyReLU(alpha=0.2))
        # ----------------------------------------------------------------
        # downsample 16x16
        model.add(SpectralNormalization(Conv2D(filters=128,
                                               kernel_size=(4, 4),
                                               kernel_initializer=init,
                                               strides=(2, 2),
                                               padding='same')))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.4))
        # ----------------------------------------------------------------
        # Downsample 8x8
        model.add(SpectralNormalization(Conv2D(filters=256,
                                               kernel_size=(4, 4),
                                               kernel_initializer=init,
                                               strides=(2, 2),
                                               padding='same')))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.4))

        # Classifier
        model.add(Flatten())

        model.add(Dense(1, activation='sigmoid'))
    else:
        # ----------------------------------------------------------------
        # Downsample 208x208
        model.add(Conv2D(filters=128,
                         kernel_size=(4, 4),
                         kernel_initializer=init,
                         strides=(2, 2),
                         padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.4))
        # ----------------------------------------------------------------

        # Downsample 104x104
        model.add(Conv2D(filters=128,
                         kernel_size=(4, 4),
                         kernel_initializer=init,
                         strides=(2, 2),
                         padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.5))

        # Downsample 52x52
        model.add(Conv2D(filters=128,
                         kernel_size=(4, 4),
                         kernel_initializer=init,
                         strides=(2, 2),
                         padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.5))

        # Downsample 52x52
        model.add(Conv2D(filters=128,
                         kernel_size=(4, 4),
                         kernel_initializer=init,
                         strides=(2, 2),
                         padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # classifier
        model.add(Flatten())
        model.add(Dropout(rate=0.4))

        model.add(Dense(1, activation='sigmoid'))
    return model


def generator():
    init = TruncatedNormal(mean=0.0, stddev=0.02)

    model = tf.keras.Sequential()
    # ----------------------------------------------------------------
    size = 26
    filters = 512
    model.add(Dense(units=size*size*filters,
                    use_bias=False,
                    kernel_initializer=init,
                    input_shape=(NOISE_DIM,)))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # ----------------------------------------------------------------

    model.add(Reshape(target_shape=(size, size, filters)))
    # ----------------------------------------------------------------

    model.add(Conv2DTranspose(filters=512,
                              kernel_size=(4, 4),
                              kernel_initializer=init,
                              strides=(2, 2),
                              padding='same',
                              use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # ----------------------------------------------------------------

    # Upsample 8x8
    model.add(Conv2DTranspose(filters=128,
                              kernel_size=(4, 4),
                              kernel_initializer=init,
                              strides=(2, 2),
                              padding='same',
                              use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.4))
    # ----------------------------------------------------------------
    # Upsample 8x8
    model.add(Conv2DTranspose(filters=128,
                              kernel_size=(4, 4),
                              kernel_initializer=init,
                              strides=(2, 2),
                              padding='same',
                              use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # Upsample 8x8
    model.add(Conv2DTranspose(filters=128,
                              kernel_size=(4, 4),
                              kernel_initializer=init,
                              strides=(2, 2),
                              padding='same',
                              use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # ----------------------------------------------------------------
    # Output: 416x416x3
    model.add(Conv2D(filters=3,
                     kernel_size=(4, 4),
                     kernel_initializer=init,
                     activation='tanh',
                     padding='same'))
    assert model.output_shape == (None, 416, 416, 3)

    return model


def smooth_positive_labels(y):
    return y - 0.3 + (np.random.random(y.shape) * 0.3)


def smooth_negative_labels(y):
    return y + np.random.random(y.shape) * 0.3


# randomly flip some labels
def noisy_labels(y, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * int(y.shape[0]))
    # choose labels to flip
    flip_ix = np.random.choice([i for i in range(int(y.shape[0]))], size=n_select)

    op_list = []
    # invert the labels in place
    # y_np[flip_ix] = 1 - y_np[flip_ix]
    for i in range(int(y.shape[0])):
        if i in flip_ix:
            op_list.append(tf.subtract(1., y[i]))
        else:
            op_list.append(y[i])

    outputs = tf.stack(op_list)
    return outputs


def plot_features(features,
                  labels,
                  examples=9,
                  disp_labels=True):
    if not math.sqrt(examples).is_integer():
        print('Please select a valid number of examples.')
        return
    imgs = []
    classes = []
    for i in range(examples):
        rnd_idx = np.random.randint(0, len(labels))
        imgs.append(features[rnd_idx, :, :, :])
        classes.append(labels[rnd_idx])

    fig, axes = plt.subplots(round(math.sqrt(examples)), round(math.sqrt(examples)), figsize=(15, 15),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.3, wspace=0.01))

    for i, ax in enumerate(axes.flat):
        if disp_labels:
            ax.title.set_text(CLASS_MAP[classes[i][0]])
        ax.imshow(imgs[i])

    plt.show()


def plot_losses(g_losses, d_losses, all_gl, all_dl, all_gl_std, all_dl_std, epochs_list, epoch):
    plt.figure(figsize=(10, 5))
    plt.title(f"Generator and Discriminator Loss - EPOCH {epoch}")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    ymax = plt.ylim()[1]
    plt.savefig('./plots/plot_line_plot_loss.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title(f"Generator and Discriminator mean Loss - EPOCH {epoch}")
    plt.errorbar(x=epochs_list, y=all_gl, yerr=all_gl_std, label="G - mean")
    plt.errorbar(x=epochs_list, y=all_dl, yerr=all_dl_std, label="D - mean")
    plt.xlabel("Epochs")
    plt.ylabel("Mean loss")
    plt.legend()
    ymax = plt.ylim()[1]
    plt.savefig('./plots/plot_line_plot_mean-loss.png')
    plt.close()

def generate_and_save_images(model, epoch, noise_dim):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(noise_dim, training=False)

    for i in range(predictions.shape[0]):
        plt.subplot(2, 2, 1 + i)
        plt.axis('off')
        plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5) / 255.)

    filename = f'./plots/generated_image_epoch_{epoch}.png'
    plt.savefig(filename)
    plt.close()


def generate_test_image(model, noise_dim=100):
    test_input = tf.random.normal([1, noise_dim])
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(5, 5))
    plt.imshow((predictions[0, :, :, 0] * 127.5 + 127.5) / 255.)
    plt.axis('off')
    plt.show()


def flip(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x)
    return x

@tf.function
def train_step(images, gen, discr):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        generated_images = gen(noise, training=True)

        real_output = discr(images, training=True)
        fake_output = discr(generated_images, training=True)

        gen_loss = generator_loss(fake_output, apply_label_smoothing=True)
        disc_loss = discriminator_loss(real_output,
                                       fake_output,
                                       apply_label_smoothing=True,
                                       label_noise=True,
                                       loss_func='gan')

        gradients_of_generator = generator_tape.gradient(gen_loss, gen.trainable_variables)
        gradients_of_discriminator = discriminator_tape.gradient(disc_loss, discr.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discr.trainable_variables))

        return gen_loss, disc_loss


def train(dataset, epochs, sample_size, seed, gen, discr):
    all_gl = np.array([])
    all_dl = np.array([])
    all_gl_mean = np.array([])
    all_dl_mean = np.array([])
    all_gl_std = np.array([])
    all_dl_std = np.array([])
    epochs_list = []
    for epoch in trange(epochs):
        g_loss = []
        d_loss = []
        start = time.time()
        i = 0
        for image_batch in dataset:
            start_time = time.time()
            gen_loss, disc_loss = train_step(images=image_batch, gen=gen, discr=discr)
            g_loss.append(gen_loss)
            d_loss.append(disc_loss)
            all_gl = np.append(all_gl, np.array([g_loss]))
            all_dl = np.append(all_dl, np.array([d_loss]))

            print(f"{i}/{sample_size}>, time pr. it: {round(time.time() - start_time, 2)}, "
                  f"Curr. g_loss: {round(float(gen_loss), 4)}, "
                  f"curr d_loss: {round(float(disc_loss), 4)}")
            i += len(image_batch)
        epochs_list.append(epoch)
        all_gl_mean = np.append(all_gl_mean, np.mean(g_loss))
        all_dl_mean = np.append(all_dl_mean, np.mean(d_loss))
        all_gl_std = np.append(all_gl_std, np.std(g_loss))
        all_dl_std = np.append(all_dl_std, np.std(d_loss))
        print('*'*50)
        print(f'Epoch: {epoch + 1} computed for {time.time() - start} sec')
        print(f'Gen_loss mean: ', np.mean(g_loss), ' std: ', np.std(g_loss))
        print(f'Disc_loss mean: ', np.mean(d_loss), ' std: ', np.std(d_loss))

        plot_losses(g_loss, d_loss, all_gl_mean, all_dl_mean, all_gl_std, all_dl_std, epochs_list, epoch+1)
        generate_and_save_images(model=gen, epoch=epoch+1, noise_dim=seed)

        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    NEW_PATH = 'IMAGE PATH'
    train_images = []
    image_names = os.listdir(NEW_PATH)
    i = 0
    for filename in tqdm(image_names):
        im = np.array(Image.open(os.path.join(NEW_PATH, filename))).astype('float32')
        im = (im - 127.5) / 127.5
        train_images.append(im)
    train_images = np.stack(np.array(train_images), axis=0).astype('float32')
    sample_size = train_images.shape[0]
    seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

    # train_images.dump('train-images-snelku.npy')
    # (train_images, train_y), (_, _) = cifar10.load_data()
    # train_images = train_images.astype('float32')
    # train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(len(train_images)).batch(BATCH_SIZE)

    generator = generator()
    discriminator = discriminator(spectral_normalization=False)

    plot_model(generator, to_file='generator-model-plot.png', show_shapes=True, show_layer_names=True)
    plot_model(discriminator, to_file='diskriminator-model-plot.png', show_shapes=True, show_layer_names=True)

    generator_optimizer = Adam(learning_rate=LEARNING_RATE_GENERATOR, beta_1=BETA_1)
    discriminator_optimizer = Adam(learning_rate=LEARNING_RATE_DISCRIMINATOR, beta_1=BETA_1)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    train(dataset=train_dataset, epochs=EPOCHS, sample_size=sample_size, seed=seed, gen=generator, discr=discriminator)
