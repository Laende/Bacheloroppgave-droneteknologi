from os import listdir
from os.path import isfile
from PIL import Image
from tqdm import tqdm
import numpy as np

import imgaug.augmenters as iaa

import os
import random
from os.path import join
import matplotlib.pyplot as plt

DATA_DIR = 'DATA DIR'
os.chdir(DATA_DIR)


IMAGE_DIR = join(DATA_DIR, 'dataset\\PascalVOC-OG-flipped\\JPEGImages')
ANN_DIR = join(DATA_DIR, 'dataset\\PascalVOC-OG-flipped\\Annotations')

NEW_IMAGE_DIR = join(DATA_DIR, 'dataset\\PascalVOC-OG-all\\JPEGImages')
NEW_ANN_DIR = join(DATA_DIR, 'dataset\\PascalVOC-OG-all\\Annotations')

NEW_IMAGE_SETS_DIR = join(DATA_DIR, 'dataset\\PascalVOC-OG-all\\ImageSets\\Main')

MAX = 2

with open(join(NEW_IMAGE_SETS_DIR, f"pipe-augmented-degrade.txt"), 'w+') as f:
    pass

image_files = [f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))]
shuffled_image_files = random.sample(image_files, len(image_files))
shuffled_image_files = random.sample(image_files, len(shuffled_image_files))[:MAX]

seq = iaa.Sequential([
    iaa.JpegCompression(compression=(99, 99))
])


for image in tqdm(shuffled_image_files):
    if len(image) > 0:
        # Åpne bildet
        im = Image.open(join(IMAGE_DIR, image))

        # Gjøre om til array med type uint8, (1920, 1080, 3)
        im = np.asarray(im).astype(np.uint8)

        # Ekspandere arrayet til å se ut som (1, 1920, 1080, 3), nødvendig siden iaa forventer en 4D matrise
        im_expand = np.expand_dims(im, 0)

        # Augmentere bildet
        augmented_image_array = seq(images=im_expand)

        # Fjerne ekstra dimensjonen satt på tidligere på første akse, resultat: (1920, 1080, 3)
        augmented_image_array = np.squeeze(augmented_image_array, axis=0)

        # Laste inn array som bilde
        augmented_image = Image.fromarray(augmented_image_array)

        # Laste inn bildet igjen fra matriseformat.
        im = Image.fromarray(im)
        im.save('im1.jpeg')
        augmented_image.save('im2.jpeg')
        fig, ax = plt.subplots(nrows=1, ncols=2)

        # Plotting
        plt.subplot(1, 2, 1)
        plt.imshow(im)

        plt.subplot(1, 2, 2)
        plt.imshow(augmented_image)
        plt.show()

