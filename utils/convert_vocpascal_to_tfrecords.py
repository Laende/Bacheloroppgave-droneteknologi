import os
import hashlib
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm
from os.path import join

DATA_DIR = '<DATADIR>'
os.chdir(DATA_DIR)

# FLAGS
split = 'train'
flags.DEFINE_string('image_set', 'pipe', 'Navnet på datasettet')
flags.DEFINE_string('data_dir', join(DATA_DIR, 'dataset/PascalVOC-OG-all/'), 'filsti til PascalVOC datasettet')
flags.DEFINE_enum('split', split, ['train', 'val'], 'train eller val split.')
flags.DEFINE_string('output_file', join(DATA_DIR, 'data/v6_og_all_tiny'
                                        + f"_{split}"
                                        + '.tfrecord'), 'Hvor datasettet lagres i .tfrecord format')
flags.DEFINE_string('classes', join(DATA_DIR, 'data/custom.names'), 'Fil med klasser')


def build_example(annotation, class_map):
    img_path = os.path.join(FLAGS.data_dir, 'JPEGImages', annotation['filename'])
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])
    depth = int(annotation['size']['depth'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []

    classes = []
    classes_text = []
    truncated = []
    views = []
    difficult_obj = []

    if 'object' in annotation:
        for obj in annotation['object']:
            difficult = bool(int(obj['difficult']))
            difficult_obj.append(int(difficult))
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_map[obj['name']])
            truncated.append(int(obj['truncated']))
            views.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example


def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def main(_argv):
    class_map = {name: idx for idx, name in enumerate(open(FLAGS.classes).read().splitlines())}
    print("----------------------------------")
    print(f"Class map: ")
    for key, value in class_map.items():
        print(f"{value}: {key}")
    print("----------------------------------")

    writer = tf.io.TFRecordWriter(FLAGS.output_file)

    image_list = open(os.path.join(FLAGS.data_dir, f'ImageSets/Main/{FLAGS.image_set}_{FLAGS.split}.txt')).read()\
        .splitlines()

    logging.info("Image list loaded: %d", len(image_list))
    for image in tqdm.tqdm(image_list):
        if len(image) > 0:
            image_name = image.split(" ")[0]
            name = image_name.split(".")[0]
            #name, _ = name.split(".")
            annotation_xml = os.path.join(FLAGS.data_dir, 'Annotations', name + '.xml')
            try:
                annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
            except FileNotFoundError as e:
                print(e)
                continue
            annotation = parse_xml(annotation_xml)['annotation']
            tf_example = build_example(annotation, class_map)
            writer.write(tf_example.SerializeToString())
        else:
            pass
    writer.close()
    logging.info("Done")


if __name__ == '__main__':
    app.run(main)
