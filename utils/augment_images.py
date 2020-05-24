from os import listdir
from os.path import isfile
from PIL import Image, ImageStat
from tqdm import tqdm
import numpy as np

import imgaug.augmenters as iaa
import xml.etree.ElementTree as ET
import os
import random
from os.path import join


def get_brightness(img):
    img = img.convert('L')
    stat = ImageStat.Stat(img)
    return stat.rms[0]


DATA_DIR = '<DATADIR>'
os.chdir(DATA_DIR)


IMAGE_DIR = join(DATA_DIR, 'dataset\\PascalVOC-OG-flipped\\JPEGImages')
ANN_DIR = join(DATA_DIR, 'dataset\\PascalVOC-OG-flipped\\Annotations')

NEW_IMAGE_DIR = join(DATA_DIR, 'dataset\\PascalVOC-OG-all\\JPEGImages')
NEW_ANN_DIR = join(DATA_DIR, 'dataset\\PascalVOC-OG-all\\Annotations')

NEW_IMAGE_SETS_DIR = join(DATA_DIR, 'dataset\\PascalVOC-OG-all\\ImageSets\\Main')

MAX = 750

with open(join(NEW_IMAGE_SETS_DIR, f"pipe-augmented-degrade.txt"), 'w+') as f:
    pass

image_files = [f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))]
shuffled_image_files = random.sample(image_files, len(image_files))
shuffled_image_files = random.sample(image_files, len(shuffled_image_files))[:MAX]

# Kjørt de her modifikasjonene en om gangen for å få 750 bilder av hver augmentasjon.
seq = iaa.Sequential([
    # For å lag noise
    # iaa.SaltAndPepper((0.12, 0.15), per_channel=0.1)
    # For å lag bildan lyser,
    # iaa.MultiplyAndAddToBrightness(mul=(1.5), add=(-30, 30))
    # Brukt den her for å gjør dem mørkan
    # iaa.Multiply((0.25, 0.35), per_channel=0.2)
    # Brukt den her for å lag store firkanta på bilda
    # iaa.CoarseDropout((0.1, 0.15), size_percent=(0.01, 0.02))
    iaa.JpegCompression(compression=(96, 98))

])

for image in tqdm(shuffled_image_files):
    if len(image) > 0:
        # Åpne bildet
        im = Image.open(join(IMAGE_DIR, image))
        image_brightness = get_brightness(im)

        # Gjøre om til array med type uint8, (1920, 1080, 3)
        im = np.asarray(im).astype(np.uint8)

        # Ekspandere arrayet til å se ut som (1, 1920, 1080, 3), nødvendig siden iaa forventer en 4D matrise
        im = np.expand_dims(im, 0)

        # Augmentere bildet
        augmented_image_array = seq(images=im)

        # Fjerne ekstra dimensjonen satt på tidligere på første akse, resultat: (1920, 1080, 3)
        augmented_image_array = np.squeeze(augmented_image_array, axis=0)

        # Laste inn array som bilde
        augmented_image = Image.fromarray(augmented_image_array)

        # Navngreier
        old_name = str(image.split('.')[0])
        old_ann_path = join(ANN_DIR, str(old_name + ".xml"))
        new_name = old_name + "_Degrade"
        new_path = join(NEW_IMAGE_DIR, new_name)

        # Lagre det nye bildet i ny path
        augmented_image.save(str(new_path + '.jpeg'), "JPEG", optimize=True)

        # Lage ny annoteringsfil
        tree = ET.parse(old_ann_path)
        tree.find('filename').text = f"{new_name}.jpeg"
        tree.write(join(NEW_ANN_DIR, f"{new_name}.xml"))

        with open(join(NEW_IMAGE_SETS_DIR, f"pipe-augmented-degrade.txt"), 'a+') as f:
            f.write(f"{new_name}.jpeg {0}\n")