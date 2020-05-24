import os
from os.path import join
import cv2
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
from yolov3_tf2.dataset import load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
import matplotlib.pyplot as plt
from os.path import isfile, join
from os import listdir

DATA_DIR = 'DATADIR'
os.chdir(DATA_DIR)

dataset = 'v6_og'
split = 'val'

IMAGE_DIR = join(DATA_DIR, 'dataset/bachelor-PascalVOC-export-LB/JPEGImages')
onlyfiles = [f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))]

flags.DEFINE_string('classes', join(DATA_DIR, 'data/custom.names'), 'path to classes file')
flags.DEFINE_integer('size', 32*10, 'resize images to')
flags.DEFINE_string('dataset', join(DATA_DIR, f'data/{dataset}_{split}.tfrecord'), 'path to dataset')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_string('output_dir', join(DATA_DIR, 'dataset/acgan/validation'), 'path to output image')
flags.DEFINE_integer('yolo_max_boxes', 100, ' YOLO max bounding boxes')
flags.DEFINE_boolean('single', True, 'Do detection on one image or validation set')


def main(_argv):
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info(f'classes loaded: {class_names}')
    dataset = load_tfrecord_dataset(file_pattern=FLAGS.dataset,
                                    class_file=FLAGS.classes,
                                    size=FLAGS.size,
                                    yolo_max_boxes=FLAGS.yolo_max_boxes)
    if FLAGS.single:
        for image, labels in dataset.take(1):
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
            scores = [scores]
            classes = [classes]

            logging.info('labels:')
            for i in range(nums[0]):
                logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                   np.array(scores[0][i]),
                                                   np.array(boxes[0][i])))

            img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
            cv2.imwrite(FLAGS.output, img)
            logging.info('output saved to: {}'.format(FLAGS.output))

    else:
        for image, labels in dataset.as_numpy_iterator():
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
            scores = [scores]
            classes = [classes]

            logging.info('labels:')
            for i in range(nums[0]):
                logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                   np.array(scores[0][i]),
                                                   np.array(boxes[0][i])))

            img = np.array(draw_outputs(image, (boxes, scores, classes, nums), class_names), dtype=np.int32)
            plt.imshow(img)
            plt.show()


if __name__ == '__main__':
    app.run(main)
