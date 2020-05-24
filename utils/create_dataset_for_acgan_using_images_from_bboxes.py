import os
from os.path import join

import cv2
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
from tqdm import tqdm
from yolov3_tf2.dataset import load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
import matplotlib.pyplot as plt

DATA_DIR = 'DATADIR'
os.chdir(DATA_DIR)

dataset = 'v6_og_all'
split = 'val'
flags.DEFINE_string('classes', join(DATA_DIR, 'data/custom.names'), 'Fil med klasser')
flags.DEFINE_integer('size', 32*13, 'Spesifisert bildestørrelse å bruke')
flags.DEFINE_string('dataset',
                    join(DATA_DIR, f'data/{dataset}_{split}.tfrecord'),
                    'Filsti til datasett i .tfrecord format')
flags.DEFINE_string('output', './output.jpg', 'sti til utbilde.')
flags.DEFINE_string('output_dir', join(DATA_DIR, 'dataset/acgan/validation'), '')
flags.DEFINE_integer('yolo_max_boxes', 100, ' YOLO maks antall bounding boxes')
flags.DEFINE_boolean('single', False, 'Deteksjon på et bilde eller hele datasettet')

def main(_argv):
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info(f'classes loaded: {class_names}')
    dataset = load_tfrecord_dataset(file_pattern=FLAGS.dataset,
                                    class_file=FLAGS.classes,
                                    size=FLAGS.size,
                                    yolo_max_boxes=FLAGS.yolo_max_boxes)
    dataset = dataset.shuffle(64)
    if FLAGS.single:
        for image, labels in dataset.take(1):
            boxes = []
            scores = []
            classes = []
            for x1, y1, x2, y2, label in labels:
                if x1 == 0 and x2 == 0:
                    continue
                print((x1, y1, x2, y2))
                boxes.append((x1, y1, x2, y2))
                scores.append(1)
                classes.append(label)
            nums = [len(boxes)]
            boxes = [boxes]
            scores = [scores]
            classes = [classes]

            logging.info('labels:')
            for i in range(nums[0]):
                left, top, right, bottom = np.ceil(np.array(boxes[0][i]) * FLAGS.size)
                print(f"left: {left}, top: {top}, right: {right}, bottom: {bottom}")
                logging.info(f'\t{class_names[int(classes[0][i])]}, {np.array(scores[0][i])}, {np.array(boxes[0][i])}')
                img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
                img = np.asarray(img[int(top):int(bottom), int(left):int(right), :], dtype=np.int32)


            img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
            cv2.imwrite(FLAGS.output, img)
            logging.info('output saved to: {}'.format(FLAGS.output))
    else:
        bounding_boxes = {'pipe_with_ice': 0,
                          'pipe': 0}
        for counter, data in tqdm(enumerate(dataset.as_numpy_iterator())):
            image, labels = data

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
            for i in range(nums[0]):
                bounding_boxes[class_names[int(classes[0][i])]] += 1
                left, top, right, bottom = np.ceil(np.array(boxes[0][i]) * FLAGS.size)
                img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cropped_img = np.asarray(img[int(top):int(bottom), int(left):int(right), :], dtype=np.int32)
                path = join(FLAGS.output_dir, class_names[int(classes[0][i])])
                if not os.path.exists(path):
                    os.makedirs(os.path.join(path))
                cv2.imwrite(join(path, f"image-{counter}.jpeg"),
                            cropped_img,
                            params=[int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    app.run(main)
