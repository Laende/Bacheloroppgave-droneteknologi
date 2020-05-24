import os
import random
from os import listdir
from os.path import join, isfile
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from decimal import Decimal

os.chdir("..")

DATA_DIR = 'DATADIR'
os.chdir(DATA_DIR)

IMAGE_DIR = join(DATA_DIR, 'dataset\\PascalVOC-OG-flipped\\JPEGImages')
ANN_DIR = join(DATA_DIR, 'dataset/PascalVOC-OG-flipped/Annotations')

NEW_IMAGE_DIR = join(DATA_DIR, 'dataset\\PascalVOC-OG-flipped-cropped\\JPEGImages')
NEW_ANN_DIR = join(DATA_DIR, 'dataset/PascalVOC-OG-flipped-cropped/Annotations')
NEW_IMAGE_SETS_DIR = join(DATA_DIR, 'dataset/PascalVOC-OG-flipped-cropped/ImageSets/Main')

onlyfiles = [f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))]


def random_crop_image_and_label(image, bboxes, p=0.5, debug=False):
    """
    crop images randomly at 50% chance but preserve all bounding boxes. the crop is guaranteed to include
    all bounding boxes.
    """
    cropped_bboxes = []
    r = tf.random.uniform([1])
    img = None
    if r < p:
        height = np.shape(image)[0]
        width = np.shape(image)[1]

        xmin_delta, ymin_delta, xmax_delta, ymax_delta = get_random_crop_delta(bboxes, height, width, debug=debug)

        bboxes1 = np.asmatrix(bboxes)
        xmin = np.asarray(bboxes1[:, 0], dtype=np.float32)
        ymin = np.asarray(bboxes1[:, 1], dtype=np.float32)
        xmax = np.asarray(bboxes1[:, 2], dtype=np.float32)
        ymax = np.asarray(bboxes1[:, 3], dtype=np.float32)

        height = np.shape(image)[0]
        width = np.shape(image)[1]

        offset_height = np.array(ymin_delta, dtype=np.int32)
        offset_width = np.array(xmin_delta, dtype=np.int32)
        target_height = np.array(np.ceil(height - ymax_delta - ymin_delta), dtype=np.int32)
        target_width = np.array(np.ceil(width - xmax_delta - xmin_delta), dtype=np.int32)

        img = image[offset_height:offset_height + target_height, offset_width:offset_width + target_width, :]

        new_height = np.shape(img)[0]
        new_width = np.shape(img)[1]

        new_xmin = ((xmin - xmin_delta) / (width - xmin_delta - xmax_delta)) * new_width
        new_ymin = (ymin - ymin_delta) / (height - ymin_delta - ymax_delta) * new_height
        new_xmax = (xmax - xmin_delta) / (width - xmin_delta - xmax_delta) * new_width
        new_ymax = (ymax - ymin_delta) / (height - ymin_delta - ymax_delta) * new_height

        cropped_bboxes = np.asarray(np.squeeze(np.stack([new_xmin,
                                                         new_ymin,
                                                         new_xmax,
                                                         new_ymax], axis=1), axis=-1), dtype=np.int32)
        if debug:
            print(f"Height: {height}")
            print(f"Width: {width}\n")

            print(f"xmin_delta: {xmin_delta}")
            print(f"ymin_delta: {ymin_delta}")
            print(f"xmax_delta: {xmax_delta}")
            print(f"ymax_delta: {ymax_delta}\n")

            print('--------')
            print(f"xmin: {new_xmin} = ({xmin} - {xmin_delta}) / ({width} - {xmin_delta} - {xmax_delta})\n")
            print(f"ymin: {new_ymin} = ({ymin} - {ymin_delta}) / ({height} - {ymin_delta} - {ymax_delta})\n")
            print(f"xmax: {new_xmax} = ({xmax} - {xmin_delta}) / ({width} - {xmin_delta} - {xmax_delta})\n")
            print(f"ymax:{new_ymax} = ({ymax} - {ymin_delta}) / ({height} - {ymin_delta} - {ymax_delta})\n")

            print('--------')
            print(f"xmin: {xmin}")
            print(f"ymin: {ymin}")
            print(f"xmax: {xmax}")
            print(f"ymax: {ymax}\n")

            print('--------')
            print(f"offset height: {offset_height}")
            print(f"Target height: {target_height}")
            print(f"Offset width: {offset_width}")
            print(f"Target width: {target_width}\n")
            print('--------')

            print(f"new_height: {new_height}")
            print(f"new_width: {new_width}\n")

            print('--------')
            print(f"Old bboxes: {bboxes}")
            print(f"New bboxes: {cropped_bboxes}\n")
    return img, cropped_bboxes

def draw_rect(img, cords, color=None):
    img = img.copy()
    cords = cords.reshape(-1, 4)
    if not color:
        color = [255, 255, 255]
    for cord in cords:
        pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])

        pt1 = pt1[0], pt1[1]
        pt2 = pt2[0], pt2[1]

        img = cv2.rectangle(img.copy(), pt1, pt2, color, int(max(img.shape[:2]) / 200))
    return img


def get_random_crop_delta(bboxes, height, width, debug=True):

    # Find the minimum and maximum values for x_min, y_min, x_max, y_max from all bounding boxes in image
    min_xmin = np.min(bboxes[..., 0])
    min_ymin = np.min(bboxes[..., 1])
    max_xmax = np.max(bboxes[..., 2])
    max_ymax = np.max(bboxes[..., 3])

    # ____________________________________
    # |         ________________         |
    # |image    |crop ______   |         |
    # |<-DELTA->|     |bbox|   |<-DELTA->|
    # |         |     |____|   |         |
    # |         |______________|         |
    # |__________________________________|

    xmin_delta = np.random.uniform(low=0, high=min_xmin)
    ymin_delta = np.random.uniform(low=0, high=min_ymin)
    xmax_delta = np.random.uniform(low=0, high=(width - max_xmax))
    ymax_delta = np.random.uniform(low=0, high=(height - max_ymax))

    if debug:
        print(f"min xmin: {min_xmin}")
        print(f"min ymin: {min_ymin}")
        print(f"max xmax: {max_xmax}")
        print(f"max ymax: {max_ymax}")

        print(f"xmin delta: {xmin_delta}")
        print(f"ymin delta: {ymin_delta}")
        print(f"xmax delta: {xmax_delta}")
        print(f"ymax delta: {ymax_delta}")

    return xmin_delta, ymin_delta, xmax_delta, ymax_delta


def crop_img_boxes(original_image, bboxes, debug=False):
    cropped_image, cropped_bboxes = random_crop_image_and_label(image=original_image, bboxes=bboxes, p=1, debug=debug)
    return cropped_image, cropped_bboxes


def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    list_with_all_boxes = []
    list_with_labels = []
    for boxes in root.iter('object'):
        name = boxes.find("name").text
        ymin, xmin, ymax, xmax = None, None, None, None
        for box in boxes.findall("bndbox"):
            ymin = int(Decimal(str(box.find("ymin").text)))
            xmin = int(Decimal(str(box.find("xmin").text)))
            ymax = int(Decimal(str(box.find("ymax").text)))
            xmax = int(Decimal(str(box.find("xmax").text)))

        list_with_labels.append(name)
        list_with_all_boxes.append([xmin, ymin, xmax, ymax])
    return np.array(list_with_all_boxes), list_with_labels


with open(os.path.join(NEW_IMAGE_SETS_DIR, f"pipe.txt"), 'w+') as f:
    pass

debug = False
for image in tqdm.tqdm(onlyfiles):
    if len(image) > 0:

        # Filen bildene er lest inn fra har formatet "bilde.jpeg 1", dette splittes i image_name og label
        # ved å bruke .split(" ")
        new_name = str(image.split(".")[0] + "_cropped")

        # Leser inn filepath til annoteringsfilen
        file = os.path.join(ANN_DIR, f"{image.split('.')[0]}.xml")

        # Original bboxes og labels leses annoteringsfilen
        original_bboxes, labels = read_content(str(file))

        # Original bilde leses inn vha cv2.imread
        original_image = cv2.imread(os.path.join(IMAGE_DIR, image))[:, :, ::-1]

        # Beskåret bilde, med tilhørende avgrensingsbokser.
        image_cropped, bboxes_cropped = crop_img_boxes(original_image, original_bboxes, debug=debug)

        # Høyde, bredde og antall kanaler for bildet.
        cropped_height, cropped_width, cropped_channels = np.shape(image_cropped)

        # Printing og sånt.
        if debug:
            print(cropped_height, cropped_width, cropped_channels)
            rect_img = draw_rect(image_cropped, bboxes_cropped)
            plt.imshow(rect_img)
            plt.show()
            continue

        # Lagre bilde i spesifisert filsti
        cv2.imwrite(os.path.join(NEW_IMAGE_DIR, f"{new_name}.jpeg"),
                    cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # Åpne annoteringsfilen for gjeldende bilde, oppdater relevant info og lagre det som ny annoteringsfil for
        # det nye beskårede bildet.
        tree = ET.parse(file)
        tree.find('filename').text = f"{new_name}.jpeg"
        tree.find('size/width').text = str(cropped_width)
        tree.find('size/height').text = str(cropped_height)
        tree.find('path').text = os.path.join(NEW_IMAGE_DIR, f"{new_name}.jpeg")

        for c, boxes in enumerate(tree.iter('object')):
            for box in boxes.findall("bndbox"):
                box.find("xmin").text = str(bboxes_cropped[c][0])
                box.find("ymin").text = str(bboxes_cropped[c][1])
                box.find("xmax").text = str(bboxes_cropped[c][2])
                box.find("ymax").text = str(bboxes_cropped[c][3])

        tree.write(os.path.join(NEW_ANN_DIR, f"{new_name}.xml"))
        with open(os.path.join(NEW_IMAGE_SETS_DIR, f"pipe.txt"), 'a+') as f:
            f.write(f"{new_name}.jpeg {0}\n")



