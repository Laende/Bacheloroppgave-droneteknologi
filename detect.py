import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
import os
from os.path import join
from tqdm import tqdm
import random

DATA_DIR = 'DATADIR'

DATA_SET = "v6_og"
tiny = False
im_size = 32*10
split = 'val'

flags.DEFINE_string('classes', join(DATA_DIR, 'data/custom.names'), 'path to classes file')
flags.DEFINE_string('weights', join(DATA_DIR, f'checkpoints/{DATA_SET}_tiny_{tiny}_im_size_{im_size}.tf'),
                    'path to weights file')
flags.DEFINE_boolean('tiny', tiny, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', im_size, 'resize images to')
flags.DEFINE_string('image_dir',
                    join(DATA_DIR, 'dataset/bachelor-PascalVOC-export-LB/JPEGImages'),
                    'path to input image directory')
flags.DEFINE_string('tfrecord', join(DATA_DIR, f'data/{DATA_SET}_{split}.tfrecord'), 'tfrecord instead of image')
flags.DEFINE_string('output', './detect_output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 2, 'number of classes in the model')
flags.DEFINE_boolean('single', True, 'Do detection on one image or validation set')
flags.DEFINE_boolean('shuffle', False, 'Shuffle dataset')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    # Load weights
    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    # Load classnames
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.single:
        if FLAGS.tfrecord:
            dataset = load_tfrecord_dataset(FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
            if FLAGS.shuffle:
                dataset = dataset.shuffle(128)
            img_raw, _label = next(iter(dataset.take(1)))
        else:
            image = random.choice(os.listdir(FLAGS.BILDE_MAPPE))
            print(f"Image chosen: {image}")
            img_raw = tf.image.decode_image(open(FLAGS.BILDE_MAPPE + "/" + image, 'rb').read(), channels=3)
        # (1080, 1920, 3) --> (1, 1080, 1920, 3)
        img = tf.expand_dims(img_raw, 0)

        # Transformerer bildet til onsket size (416, 416, 3)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        logging.info('detections:')
        for i in range(nums[0]):
            logging.info(f'\t{class_names[int(classes[0][i])]}, {np.array(scores[0][i])}, {np.array(boxes[0][i])}')

        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(FLAGS.output, img)
        logging.info('output saved to: {}'.format(FLAGS.output))

    else:
        if FLAGS.tfrecord:
            dataset = load_tfrecord_dataset(FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
            if FLAGS.shuffle:
                dataset = dataset.shuffle(512)
            dataset = dataset.as_numpy_iterator()

            times = []
            for img_raw, _label in tqdm(dataset):
                img = transform_images(img_raw, FLAGS.size)

                t1 = time.time()
                boxes, scores, classes, nums = yolo(img)
                t2 = time.time()
                times.append(t2-t1)

            mean_times = np.mean(times)
            print(f"Mean detection time for a total of {len(dataset)} was {mean_times}s")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
