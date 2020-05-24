import os
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
from tqdm import tqdm
from yolov3_tf2.dataset import load_tfrecord_dataset
from decimal import Decimal
from os.path import join
import cv2

from yolov3_tf2.utils import draw_outputs

os.chdir("..")

DATA_DIR = 'C:/Users/gieri/OneDrive/Skole/UiT/Droneteknologi/' \
                              '6. semester/Bacheloroppgave/Hovedprosjekt/Data'
data_set_name = 'v6_og_all'
split = 'val'
im_size = 32 * 16

if data_set_name == 'v6_og':
    data_set_folder = 'PascalVOC-OG'
elif data_set_name == 'v6_og_flipped':
    data_set_folder = 'PascalVOC-OG-flipped'
elif data_set_name == 'v6_og_all':
    data_set_folder = 'PascalVOC-OG-all'
else:
    data_set_folder = 'None'

flags.DEFINE_string('classes', join(DATA_DIR, 'data/custom.names'), 'path to classes file')
flags.DEFINE_integer('size', im_size, 'resize images to')
flags.DEFINE_string('dataset', join(DATA_DIR, f'data/{data_set_name}_{split}.tfrecord'), 'path to dataset')
flags.DEFINE_boolean('save_images', True, 'Save image detections or detection text files')

GROUND_TRUTH_DIR = join(DATA_DIR, f'dataset/{data_set_folder}/groundtruths')

def main(_argv):
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info(f'classes loaded: {class_names}')
    dataset = load_tfrecord_dataset(file_pattern=FLAGS.dataset,
                                    class_file=FLAGS.classes,
                                    size=FLAGS.size)
    dataset = dataset.as_numpy_iterator()
    counter = 0
    for val_data in tqdm(dataset):
        counter += 1
        filename = f"image_size_{FLAGS.size}/image-{counter}.txt"
        image_name = f"image_size_{FLAGS.size}/images/image-{counter}.jpeg"
        image, labels = val_data

        boxes = []
        scores = []
        classes = []
        for x1, y1, x2, y2, label in labels:
            if x1 == 0 and x2 == 0:
                continue
            boxes.append((x1, y1, x2, y2))
            scores.append(1)
            classes.append(label)
        nums = [len(boxes)]
        boxes = [boxes]
        classes = [classes]

        if FLAGS.save_images:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names, text=True, obj=False)
            cv2.imwrite(os.path.join(GROUND_TRUTH_DIR, image_name), img)
        else:
            # Clear file
            open(os.path.join(GROUND_TRUTH_DIR, filename), "w").close()
            for i in range(nums[0]):
                left, top, right, bottom = np.array(boxes[0][i]) * FLAGS.size
                with open(os.path.join(GROUND_TRUTH_DIR, filename), 'a+') as f:
                    f.write(f"{class_names[int(classes[0][i])]} "
                            f"{round(Decimal(str(left)), 2)} "
                            f"{round(Decimal(str(top)), 2)} "
                            f"{round(Decimal(str(right)), 2)} "
                            f"{round(Decimal(str(bottom)), 2)}\n")


if __name__ == '__main__':
    app.run(main)
