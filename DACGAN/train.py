import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from DACGAN.dacgan import Acgan

# Parametere for trening
IMG_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 10
BATCH_SIZE = 256
EPOCHS = 100000


if __name__ == '__main__':
    acgan = Acgan(img_size=IMG_SIZE, num_classes=NUM_CLASSES, channels=CHANNELS, create_new=False, cifar10=True)
    acgan.train(epochs=EPOCHS, batch_size=BATCH_SIZE, sample_interval=250)
