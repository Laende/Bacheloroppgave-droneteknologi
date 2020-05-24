import glob
import imageio
from tqdm import tqdm
import tensorflow as tf


anim_file = 'snelku.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('generated*.png')
  filenames = sorted(filenames)
  last = -1
  for i, filename in tqdm(enumerate(filenames)):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
