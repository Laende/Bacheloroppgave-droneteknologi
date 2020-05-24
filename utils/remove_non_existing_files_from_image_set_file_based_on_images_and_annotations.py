from os import listdir, remove
from os.path import join, isfile

IMAGE_DIR = 'IMAGE DIR'
ANN_DIR = 'ANNOTATION DIR'
IMAGE_SETS_DIR = 'IMAGESETS DIR'

image_files = [f.split('.')[0] for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))]
annotation_files = [f.split('.')[0] for f in listdir(ANN_DIR) if isfile(join(ANN_DIR, f))]

non_duplicates = list(set(annotation_files) - set(image_files))

for f in non_duplicates:
    fname = str(f.rstrip() + '.xml')
    fpath = join(ANN_DIR, fname)
    if isfile(fpath):
        remove(fpath)

with open(join(IMAGE_SETS_DIR, 'pipe.txt')) as oldfile, open(join(IMAGE_SETS_DIR, 'pipe_new.txt'), 'w') as newfile:
    for line in oldfile:
        clean = True
        for file in non_duplicates:
            if file in line:
                clean = False
        if clean:
            newfile.write(line)

