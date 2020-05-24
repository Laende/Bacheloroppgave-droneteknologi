from os import listdir
from os.path import isfile, join
import os
import cv2

os.chdir("..")

BILDE_MAPPE = '<BILDEMAPPE>'
# Les bare filnavnene i mappa
onlyfiles = [f for f in listdir(BILDE_MAPPE) if isfile(join(BILDE_MAPPE, f))]


for image in onlyfiles:
    # Full "path" til bildefilen
    path = join(BILDE_MAPPE, image)

    # Lese bildet gitt i pathen over
    original_image = cv2.imread(path)

    # Konvertere lest bilde til gråskala bilde
    grayImage = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # hvis bildet inneholder flere enn 2 kanaler
    if len(grayImage.shape) > 2:
        # Lagre bildet med 3 kanaler (høyde bredde 1) med parameter JPEG kvalitet : 100%
        cv2.imwrite(path, grayImage[:, :, 1], params=[int(cv2.IMWRITE_JPEG_QUALITY), 100])

