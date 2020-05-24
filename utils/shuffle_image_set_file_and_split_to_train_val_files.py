import random
from tqdm import tqdm
from os.path import join

DATA_DIR = 'DATADIR'

IMAGE_SETS_DIR = join(DATA_DIR, 'dataset/PascalVOC-OG-all/ImageSets/Main')

with open(join(IMAGE_SETS_DIR, 'pipe.txt')) as f:
    lines = f.readlines()

f_train_out = open(join(IMAGE_SETS_DIR, 'pipe_train.txt'), 'w')
f_val_out = open(join(IMAGE_SETS_DIR, 'pipe_val.txt'), 'w')

# Shuffle flere ganger
shuffled_lines = random.sample(lines, len(lines))
shuffled_lines = random.sample(shuffled_lines, len(shuffled_lines))
shuffled_lines = random.sample(shuffled_lines, len(shuffled_lines))
shuffled_lines = random.sample(shuffled_lines, len(shuffled_lines))

for line in tqdm(shuffled_lines):
    # Tilfeldig tall mellom 0 og 1
    r = random.random()

    # om r er mindre enn 0.8 så...
    if r < 0.8:
        f_train_out.write(line)
    # Ellers...
    else:
        f_val_out.write(line)

# Lukk filene
f_train_out.close()
f_val_out.close()
