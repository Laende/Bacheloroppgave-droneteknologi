from os import listdir
from os.path import join, isfile
from tqdm import tqdm
from shutil import copyfile

IMAGE_DIR = 'IMAGE DIR'
ANN_DIR = 'ANNOTATION DIR'
NEW_ANN_DIR = 'NEW ANNOTATION DIR'

image_files = [f.split('.')[0] for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))]
annotation_files = [f.split('.')[0] for f in listdir(ANN_DIR) if isfile(join(ANN_DIR, f))]

for image in tqdm(image_files):
    for annotation_file in annotation_files:
        if image == annotation_file:
            old_file_path = join(ANN_DIR, f'{annotation_file}.xml')
            new_file_path = join(NEW_ANN_DIR, f'{annotation_file}.xml')
            copyfile(old_file_path, new_file_path)

