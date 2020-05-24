import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from requests import get
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger)
from tensorflow.keras.utils import plot_model
from yolov3_tf2.utils import load_darknet_weights
import yolov3_tf2.dataset as dataset
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks)
from yolov3_tf2.utils import freeze_all

# Set working directory
DATA_DIR = 'DATADIR'
os.chdir(DATA_DIR)

# FLAGS
data_set = "v6_og_all"
flags.DEFINE_string('dataset', './data/' + data_set + '_train.tfrecord', 'path to dataset')
flags.DEFINE_string('val_dataset', './data/' + data_set + '_val.tfrecord', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('tiny_weights', './data/yolov3-tiny.weights', 'path to tiny weights file')
flags.DEFINE_string('weights', './data/yolov3.weights', 'path to weights file')
flags.DEFINE_string('weights_tf_format', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('weights_tf_format_tiny', './checkpoints/yolov3-tiny.tf',
                    'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov3.tf', 'path to output')
flags.DEFINE_string('classes', './data/custom.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'darknet',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 32*16, 'image size')
flags.DEFINE_integer('epochs', 500, 'number of epochs')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('buffer_size', 10000, 'Shuffler buffer size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 2, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', 80, 'specify num class for `weights` file if different, '
                                                'useful in transfer learning with different number of classes')

def check_weighs_exist(tiny=False):
    if not tiny:
        is_file = os.path.isfile(f"{FLAGS.weights_tf_format}.index")
        if not is_file:
            is_yolov3_weights_file = os.path.isfile(f"{FLAGS.weights}")
            if not is_yolov3_weights_file:
                logging.info("Weights file doesn't exist, "
                             "downloading YOLOv3 weights and converting to tfrecords format..")
                url = "https://pjreddie.com/media/files/yolov3.weights"
                file_name = FLAGS.weights
                download_weights(url=url, file_name=file_name)
                convert_weights_to_tf()
            else:
                logging.info("Weights file exist, converting to tfrecords format..")
                convert_weights_to_tf()
    else:
        is_file = os.path.isfile(f"{FLAGS.weights_tf_format_tiny}.index")
        if not is_file:
            is_yolov3_weights_file = os.path.isfile(FLAGS.tiny_weights)
            if not is_yolov3_weights_file:
                logging.info("Weights file doesn't exist, "
                             "downloading YOLOv3 weights and converting to tfrecords format..")
                url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
                file_name = FLAGS.tiny_weights
                download_weights(url=url, file_name=file_name)
                convert_weights_to_tf()
            else:
                logging.info("Weights file exist, converting to tfrecords format..")
                convert_weights_to_tf()


def download_weights(url, file_name):
    with open(file_name, "wb") as f:
        logging.info(f"Downloading: {file_name.split('/')[-1]}")
        response = get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=512):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()
            logging.info("Download complete.")


def convert_weights_to_tf():
    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=80)
        load_darknet_weights(model=yolo, weights_file=FLAGS.tiny_weights, tiny=FLAGS.tiny)
        logging.info('weights loaded')

        img = np.random.random((1, 320, 320, 3)).astype(np.float32)
        output = yolo(img)
        logging.info('sanity check passed')

        yolo.save_weights('./checkpoints/yolov3-tiny.tf')
        logging.info('weights saved')
    else:
        yolo = YoloV3(classes=80)

        load_darknet_weights(model=yolo, weights_file=FLAGS.weights, tiny=FLAGS.tiny)
        logging.info('weights loaded')

        img = np.random.random((1, 320, 320, 3)).astype(np.float32)
        output = yolo(img)
        logging.info('sanity check passed')

        yolo.save_weights(FLAGS.output)
        logging.info('weights saved')

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    check_weighs_exist(tiny=FLAGS.tiny)

    if FLAGS.tiny:
        model = YoloV3Tiny(
            FLAGS.size,
            training=True,
            classes=FLAGS.num_classes
        )
        model.summary()
        plot_model(model, to_file='yoloV3Tiny-model-plot.png', show_shapes=True, show_layer_names=True)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(
            FLAGS.size,
            training=True,
            classes=FLAGS.num_classes
        )
        model.summary()
        plot_model(model, to_file='yoloV3-model-plot.png', show_shapes=True, show_layer_names=True)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    # Load the dataset
    train_dataset = dataset.load_fake_dataset()

    if FLAGS.dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            file_pattern=FLAGS.dataset,
            class_file=FLAGS.classes,
            size=FLAGS.size
        )
    # Shuffle the dataset
    train_dataset = train_dataset.shuffle(buffer_size=FLAGS.buffer_size, reshuffle_each_iteration=True)
    train_dataset_length = [i for i, _ in enumerate(train_dataset)][-1] + 1
    print(f"Dataset for training consists of {train_dataset_length} images.")

    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (dataset.transform_images(x, FLAGS.size),
                                                    dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size))).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_fake_dataset()
    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size)

    val_dataset_length = [i for i, _ in enumerate(val_dataset)][-1] + 1
    print(f"Dataset for validation consists of {val_dataset_length} images.")
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (dataset.transform_images(x, FLAGS.size),
                                                dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size))).repeat()

    # Configure the model for transfer learning
    if FLAGS.transfer == 'none':
        pass  # Nothing to do
    elif FLAGS.transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes
        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                size=FLAGS.size,
                training=True,
                classes=FLAGS.weights_num_classes or FLAGS.num_classes)
            model_pretrained.load_weights(FLAGS.weights_tf_format_tiny)

        else:
            model_pretrained = YoloV3(
                size=FLAGS.size,
                training=True,
                classes=FLAGS.weights_num_classes or FLAGS.num_classes)
            model_pretrained.load_weights(FLAGS.weights_tf_format)

        if FLAGS.transfer == 'darknet':
            # Set yolo darknet layer weights to the loaded pretrained model weights
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            # Freeze these layers
            freeze_all(model.get_layer('yolo_darknet'))

        elif FLAGS.transfer == 'no_output':
            for i in model.layers:
                if not i.name.startswith('yolo_output'):
                    i.set_weights(model_pretrained.get_layer(
                        i.name).get_weights())
                    freeze_all(i)

    else:
        # All other transfer require matching classes
        if FLAGS.tiny:
            model.load_weights(FLAGS.weights_tf_format_tiny)
        else:
            model.load_weights(FLAGS.weights_tf_format)
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.transfer == 'frozen':
            # freeze everything
            freeze_all(model)

    # Use the Adam optimizer with the specified learning rate
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)

    # YoloLoss function
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes) for mask in anchor_masks]

    if FLAGS.mode == 'eager_tf':
        print(f"Mode is: {FLAGS.mode}")
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)

                    regularization_loss = tf.reduce_sum(model.losses)

                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                logging.info(f"epoch_{epoch}_train_batch_{batch},"
                             f"{total_loss.numpy()},"
                             f"{list(map(lambda x: np.sum(x.numpy()), pred_loss))}")
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                logging.info("{}_val_{}, {}, {}".format(
                    epoch,
                    batch,
                    total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()),
                             pred_loss)))
                )
                avg_val_loss.update_state(total_loss)

            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights(f'checkpoints/{data_set}_tiny_{FLAGS.tiny}_im_size_{FLAGS.size}.tf')
    else:
        print(f"Compiling the model")
        model.compile(
            optimizer=optimizer,
            loss=loss,
            run_eagerly=(FLAGS.mode == 'eager_fit'),
            metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss',
                      patience=125,
                      verbose=1),
        ReduceLROnPlateau(monitor='val_loss',
                          verbose=1,
                          factor=0.90,
                          min_lr=0,
                          patience=20,
                          mode="auto"),
        ModelCheckpoint(
            str(f'checkpoints/{data_set}_tiny_{FLAGS.tiny}_im_size_{FLAGS.size}.tf'),
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
            mode="auto",
        ),
        TensorBoard(log_dir='logs'),
        CSVLogger(f'checkpoints/logs/{data_set}_tiny_{FLAGS.tiny}_im_size_{FLAGS.size}',
                  separator=',')
    ]
    history = model.fit(train_dataset,
                        epochs=FLAGS.epochs,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=val_dataset,
                        steps_per_epoch=np.ceil(train_dataset_length / FLAGS.batch_size),
                        validation_steps=np.ceil(val_dataset_length / FLAGS.batch_size))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
