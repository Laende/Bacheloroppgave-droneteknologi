import time
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
import os
from tqdm import tqdm
from os.path import join
from decimal import Decimal
import cv2
from yolov3_tf2.utils import draw_outputs
os.chdir("..")

DATA_DIR = 'C:/Users/gieri/OneDrive/Skole/UiT/Droneteknologi/' \
                              '6. semester/Bacheloroppgave/Hovedprosjekt/Data'
data_set_name = 'v6_og_all'
split = 'val'
im_size = 32 * 16

score_threshold = 0.5
tiny = False

if data_set_name == 'v6_og':
    data_set_folder = 'PascalVOC-OG'
elif data_set_name == 'v6_og_flipped':
    data_set_folder = 'PascalVOC-OG-flipped'
elif data_set_name == 'v6_og_all':
    data_set_folder = 'PascalVOC-OG-all'
else:
    data_set_folder = 'None'

flags.DEFINE_string('classes', join(DATA_DIR, 'data/custom.names'), 'Fil med klasser')
flags.DEFINE_integer('size', im_size, 'Spesifisert bildestørrelse å bruke')
flags.DEFINE_string('dataset',
                    join(DATA_DIR, f'data/{data_set_name}_{split}.tfrecord'),
                    'Filsti til datasett i .tfrecord format')
flags.DEFINE_string('weights', join(DATA_DIR, f'checkpoints/{data_set_name}_tiny_{tiny}_im_size_{im_size}.tf'),
                    'path to weights file')
flags.DEFINE_boolean('tiny', tiny, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('output', './detect_output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 2, 'number of classes in the model')
flags.DEFINE_boolean('save_images', True, 'Save image detections or detection text files')

DETECTIONS_DIR = join(DATA_DIR, f'dataset/{data_set_folder}/detections')


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
    logging.info(f'classes loaded: {class_names}')
    dataset = load_tfrecord_dataset(file_pattern=FLAGS.dataset,
                                    class_file=FLAGS.classes,
                                    size=FLAGS.size)
    dataset = dataset.as_numpy_iterator()
    times = []
    counter = 0
    for val_data in tqdm(dataset):
        counter += 1
        file_name = f"image_size_{im_size}/tiny/score_threshold_{score_threshold}/image-{counter}.txt"
        image_name = f"image_size_{im_size}/score_threshold_{score_threshold}/images/image-{counter}.jpeg"
        img_raw, _label = val_data

        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        times.append([np.subtract(t2, t1)])
        logging.info('time: {}'.format(t2 - t1))

        if FLAGS.save_images:
            img = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names, text=True, color=(0, 0, 255), obj=False)
            cv2.imwrite(os.path.join(DETECTIONS_DIR, image_name), img)

        else:
            open(os.path.join(DETECTIONS_DIR, file_name), "w").close()
            for i in range(nums[0]):
                left, top, right, bottom = np.array(boxes[0][i]) * FLAGS.size
                with open(os.path.join(DETECTIONS_DIR, file_name), 'a+') as f:
                    f.write(f"{class_names[int(classes[0][i])]} "
                            f"{np.array(scores[0][i])} "
                            f"{round(Decimal(str(left)), 2)} "
                            f"{round(Decimal(str(top)), 2)} "
                            f"{round(Decimal(str(right)), 2)} "
                            f"{round(Decimal(str(bottom)), 2)}\n")

            print(f"Mean detection time for dataset: {data_set_name} "
                  f"with image size: {im_size} "
                  f"and score threshold: {score_threshold} is: "
                  f"{round(np.mean(times), 3)}, fps: {round(1/(np.mean(times)), 2)}")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
