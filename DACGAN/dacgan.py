import os
import pickle
from collections import defaultdict

from DACGAN.utils import load_dataset
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, LeakyReLU, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tqdm import trange
import cv2
import numpy as np


class Acgan():
    def __init__(self, img_size, channels, num_classes, create_new, cifar10=False):

        self.img_size = img_size
        self.channels = channels
        self.img_shape = (self.img_size, self.img_size, self.channels)

        self.num_classes = num_classes
        self.latent_dim = 100
        self.create_new = create_new
        # Bruke cifar10 datasettet eller ikke.
        self.cifar10 = cifar10
        self.gen_history = []
        self.label_history = []

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Bygg og kompiler discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Bygg generator
        self.generator = self.build_generator()

        # Generatoren tar imot støy og klassen som inndata,
        # og genererer et bilde av gitt klasse.
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        label_tensor = Input(shape=(int(self.img_size / 2), int(self.img_size / 2), self.num_classes), dtype='float32')
        img = self.generator([noise, label, label_tensor])

        # For den kombinerte modellen vil vi bare trene generatoren.
        self.discriminator.trainable = False

        # Diskriminatoren tar inn det genererte bildet som inngangsdata og forteller om den er "fake" eller ikke.
        # I tillegg gir den ut klassen til bildet.
        valid, target_label = self.discriminator(img)

        self.combined = Model([noise, label, label_tensor], [valid, target_label])
        self.combined.compile(loss=losses, optimizer=optimizer)

    def build_generator(self):
        d1 = int(self.img_size / 8)

        model = Sequential()

        # Første blokk (Dense)
        model.add(Dense(384 * d1 * d1, input_dim=self.latent_dim, kernel_initializer='he_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((d1, d1, 384)))  # size:(d1,d1,256)

        # Andre blokk (Konvolusjonal)
        model.add(UpSampling2D(interpolation='bilinear'))  # size: (2*d1,2*d1,256)
        model.add(Conv2D(filters=192,
                         kernel_size=3,
                         padding="same",
                         kernel_initializer='he_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # Tredje blokk (Konvolusjonal)
        model.add(UpSampling2D(interpolation='bilinear'))  # size: (4*d1,4*d1,256)

        model2 = Sequential()  # size: (4*d1,4*d1,256+num_classes)
        model2.add(Conv2D(filters=96,
                          kernel_size=3,
                          padding="same",  # size: (4*d1,4*d1,128)
                          input_shape=(4 * d1, 4 * d1, 96*2 + self.num_classes),
                          kernel_initializer='he_normal'))
        model2.add(BatchNormalization(momentum=0.8))
        model2.add(LeakyReLU(alpha=0.2))

        # Fjerde blokk (Konvolusjonal)
        model2.add(UpSampling2D(interpolation='bilinear'))  # size: (img_size,img_size,128)
        model2.add(Conv2D(filters=128,
                          kernel_size=3,
                          padding="same",
                          kernel_initializer='he_normal'))  # size: (img_size,img_size,64)
        model2.add(BatchNormalization(momentum=0.8))
        model2.add(LeakyReLU(alpha=0.2))

        # Konvolusjonalt lag + aktivering (tanh)
        model2.add(Conv2D(filters=self.channels,
                          kernel_size=3,
                          padding='same'))  # size: (img_size,img_size,3)
        model2.add(Activation("tanh"))

        model2.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_tensor = Input(shape=(4 * d1, 4 * d1, self.num_classes), dtype='float32')

        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))
        model_input = multiply([noise, label_embedding])
        r = model(model_input)

        merged = concatenate([r, label_tensor])

        img = model2(merged)

        return Model([noise, label, label_tensor], img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(filters=16,
                         kernel_size=3,
                         strides=2,
                         input_shape=self.img_shape,
                         padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(filters=32,
                         kernel_size=3,
                         strides=1,
                         padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.5))

        model.add(Conv2D(filters=64,
                         kernel_size=3,
                         strides=2,
                         padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.5))

        model.add(Conv2D(filters=128,
                         kernel_size=3,
                         strides=1,
                         padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.5))

        model.add(Conv2D(filters=256,
                         kernel_size=3,
                         strides=2,
                         padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(filters=512,
                         kernel_size=3,
                         strides=1,
                         padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.img_shape)

        features = model(img)

        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size, replay=True, sample_interval=50):

        # Last inn datasett
        X_train, y_train, number_of_classes = load_dataset(self.img_size, self.create_new, self.cifar10)


        if number_of_classes != self.num_classes:
            raise ValueError("The number of classes found is " + str(number_of_classes) +
                             " but the number of classes specified is " + str(self.num_classes) +
                             "\n Maybe there was some empty folder?")

        train_history = defaultdict(list)
        test_history = defaultdict(list)
        # Adversarial ground truths
        valid_o = np.ones((batch_size, 1))
        fake_o = np.zeros((batch_size, 1))
        pbar = trange(epochs)
        for epoch in pbar:
            # Label smoothing:
            valid = self.label_smoothing(valid_o)
            fake = self.label_smoothing(fake_o)

            # ---------------------
            #  Tren discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.randint(0, self.num_classes, (batch_size, 1))
            label_tensor = self.get_label_tensor(sampled_labels, batch_size)
            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels, label_tensor])
            # Replay:
            if replay:
                if epoch > 100 and epoch % 10:
                    self.gen_history.append(gen_imgs[0])
                    self.label_history.append(sampled_labels[0])
                if epoch > 200:
                    gen_imgs, sampled_labels = self.add_replays(gen_imgs, sampled_labels, batch_size)
                    label_tensor = self.get_label_tensor(sampled_labels, batch_size)

            # Image labels. 0-9
            img_labels = y_train[idx]

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            train_history['discriminator'].append(d_loss)
            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels, label_tensor], [valid, sampled_labels])
            train_history['generator'].append(g_loss)
            # Plot the progress
            pbar.set_description("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
            epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                curr_dir = os.path.dirname(os.path.abspath(__file__))
                pickle.dump({'train': train_history, 'test': test_history},
                            open(os.path.join(curr_dir,'logs/DACGAN-history.pkl'), 'wb'))

                self.save_model(epoch)
                self.sample_images(epoch, batch_size)

    def sample_images(self, epoch, batch_size, class_img=5):  # TODO: Change 4 batch
        noise = np.random.normal(0, 1, (1, 100))
        sampled_labels = np.array([class_img])
        label_tensor = self.get_label_tensor(sampled_labels, 1)
        gen_imgs = self.generator.predict([noise, sampled_labels, label_tensor])

        # Rescale image to 0 - 255
        gen_imgs = 255 * (0.5 * gen_imgs + 0.5)
        gen_imgs = gen_imgs.astype(np.int64)
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        cv2.imwrite(os.path.join(curr_dir, "images/" + str(epoch) + ".jpg"), gen_imgs[0])

    def save_model(self, epoch):
        def save(model, model_name, epoch):
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(curr_dir, f"saved_model/{model_name}_{epoch}.json")
            weights_path = os.path.join(curr_dir, f"saved_model/{model_name}_{epoch}_weights.hdf5")
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            with open(options['file_arch'], 'w') as f:
                f.write(json_string)
            f.close()
            model.save_weights(options['file_weight'])

        save(self.generator, "generator", epoch=epoch)
        save(self.discriminator, "discriminator", epoch=epoch)

    def label_smoothing(self, vector, max_dev=0.2):
        d = max_dev * np.random.rand(vector.shape[0], vector.shape[1])
        if vector[0][0] == 0:
            return vector + d
        else:
            return vector - d

    def add_replays(self, gen_imgs, sampled_labels, epochs, proportion=0.2):
        """
        Substitute randomly a portion of the newly generated images with some
        older (generated) ones
        """
        if len(self.gen_history) > epochs + 1:
            self.gen_history = self.gen_history[1:]
            self.label_history = self.label_history[1:]

        n = int(gen_imgs.shape[0] * proportion)
        n = min(n, len(self.label_history))
        idx_gen = np.random.randint(0, gen_imgs.shape[0], n)
        idx_hist = np.random.randint(0, len(self.gen_history), n)
        for i_g, i_h in zip(idx_gen, idx_hist):
            gen_imgs[i_g] = self.gen_history[i_h]
            sampled_labels[i_g] = self.label_history[i_h]
        return gen_imgs, sampled_labels

    def get_label_tensor(self, label, batch_size):
        shape = (batch_size, int(self.img_size / 2), int(self.img_size / 2), self.num_classes)
        t = np.zeros(shape=shape, dtype='float32')
        for i in range(batch_size):
            idx = int(label[i])
            t[i, :, :, idx] = 1

        return t
