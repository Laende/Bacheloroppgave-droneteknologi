import os
from os import listdir
from os.path import join, isfile
import xml.etree.ElementTree as ET

ANN_DIR = 'ANNOTATIONS DIR'
GROUND_TRUTH_DIR = 'GROUND TRUTH DIR'

onlyfiles = [f for f in listdir(ANN_DIR) if isfile(join(ANN_DIR, f))]


def read_write_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text.split('.')[0]
    with open(os.path.join(GROUND_TRUTH_DIR, f"{filename}.txt"), 'w+') as f:
        pass

    with open(os.path.join(GROUND_TRUTH_DIR, f"{filename}.txt"), 'a+') as f:
        for boxes in root.iter('object'):
            name = boxes.find("name").text
            ymin, xmin, ymax, xmax = None, None, None, None
            for box in boxes.findall("bndbox"):
                ymin = str(box.find("ymin").text)
                xmin = str(box.find("xmin").text)
                ymax = str(box.find("ymax").text)
                xmax = str(box.find("xmax").text)
            f.write(f"{name} {xmin} {ymin} {xmax} {ymax}\n")


for xml_file in onlyfiles:
    xml_file_path = join(ANN_DIR, xml_file)
    read_write_content(xml_file_path)
