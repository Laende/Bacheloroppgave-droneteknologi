import os
from os import listdir
from os.path import join, isfile
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import tqdm

DATA_DIR = 'C:/Users/gieri/OneDrive/Skole/UiT/Droneteknologi/6. semester/Bacheloroppgave/Hovedprosjekt/Data'
os.chdir(DATA_DIR)

IMAGE_DIR = join(DATA_DIR, 'dataset\\PascalVOC-OG\\JPEGImages')
ANN_DIR = join(DATA_DIR, 'dataset/PascalVOC-OG/Annotations')

NEW_IMAGE_DIR = join(DATA_DIR, 'dataset\\PascalVOC-OG-flipped\\JPEGImages')
NEW_ANN_DIR = join(DATA_DIR, 'dataset/PascalVOC-OG-flipped/Annotations')
NEW_IMAGE_SETS_DIR = join(DATA_DIR, 'dataset/PascalVOC-OG-flipped/ImageSets/Main')

onlyfiles = [f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))]


def flip_img_bboxes(original_image, bboxes):
    original_image_copy = original_image.copy()

    img_center = np.array(original_image_copy.shape[:2], )[::-1] / 2
    img_center = np.hstack((img_center, img_center))
    original_image_copy = original_image_copy[:, ::-1, :]

    img_center = img_center.astype('float32')

    flipped_bboxes = bboxes.copy()
    flipped_bboxes = flipped_bboxes.astype('float32')
    flipped_bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - flipped_bboxes[:, [0, 2]])

    box_w = abs(flipped_bboxes[:, 0] - flipped_bboxes[:, 2])

    flipped_bboxes[:, 0] -= box_w
    flipped_bboxes[:, 2] += box_w

    return original_image_copy, flipped_bboxes


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


def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    list_with_all_boxes = []
    list_with_labels = []
    for boxes in root.iter('object'):
        name = boxes.find("name").text
        ymin, xmin, ymax, xmax = None, None, None, None
        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        list_with_labels.append(name)
        list_with_all_boxes.append([xmin, ymin, xmax, ymax])
    return np.array(list_with_all_boxes), list_with_labels


with open(os.path.join(NEW_IMAGE_SETS_DIR, f"pipe.txt"), 'w+') as f:
    pass

for image in tqdm.tqdm(onlyfiles):
    if len(image) > 0:
        # Nytt bildenavn

        # Filen bildene er lest inn fra har formatet "bilde.jpeg 1", dette splittes i image_name og label
        # ved å bruke .split(" ")
        new_name = str(image.split(".")[0] + "_flipped")

        # Leser inn filepath til annoteringsfilen
        file = os.path.join(ANN_DIR, f"{image.split('.')[0]}.xml")

        # Original bboxes og labels leses annoteringsfilen
        original_bboxes, labels = read_content(str(file))

        # Original bilde leses inn vha cv2.imread
        original_image = cv2.imread(os.path.join(IMAGE_DIR, image))[:, :, ::-1]

        # Får ut speilvendt bilde med tilhørende avgrensingsbokser ved å sende originalbilde med tilhørende
        # avgrensingsbokser inn i funksjonen.
        image_flipped, bboxes_flipped = flip_img_bboxes(original_image, original_bboxes)

        # Lagre det speilvendte bildet i spesifisert filsti
        cv2.imwrite(os.path.join(NEW_IMAGE_DIR, f"{new_name}.jpeg"),
                    cv2.cvtColor(image_flipped, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # Åpne annoteringsfilen for gjeldende bilde, oppdater relevant info og lagre det som ny annoteringsfil for
        # det nye speilvendte bildet.
        tree = ET.parse(file)
        tree.find('filename').text = f"{new_name}.jpeg"
        tree.find('path').text = os.path.join(NEW_IMAGE_DIR, f"{new_name}.jpeg")
        for c, boxes in enumerate(tree.iter('object')):
            ymin, xmin, ymax, xmax = None, None, None, None
            for box in boxes.findall("bndbox"):
                box.find("xmin").text = str(bboxes_flipped[c][0])
                box.find("ymin").text = str(bboxes_flipped[c][1])
                box.find("xmax").text = str(bboxes_flipped[c][2])
                box.find("ymax").text = str(bboxes_flipped[c][3])

        tree.write(os.path.join(NEW_ANN_DIR, f"{new_name}.xml"))
        with open(os.path.join(NEW_IMAGE_SETS_DIR, f"pipe.txt"), 'a+') as f:
            f.write(f"{new_name}.jpeg {0}\n")



