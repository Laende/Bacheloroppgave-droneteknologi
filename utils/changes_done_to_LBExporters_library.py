import os
import json
import logging
from shapely import wkt
import requests
from PIL import Image
import numpy as np
from .pascal_voc_writer import Writer as PascalWriter
from tqdm import tqdm


class UnknownFormatError(Exception):
    """Exception raised for unknown label_format"""

    def __init__(self, label_format):
        self.message = ("Provided label_format '{}' is unsupported"
                        .format(label_format))


def from_json(labeled_data,
              annotations_output_dir,
              images_output_dir,
              image_sets_dir,
              label_format='WKT',
              database='unknown',
              use_local=False,
              local_image_dir=''):
    """Convert Labelbox JSON export to Pascal VOC format.

    Args:
        labeled_data (str): File path to Labelbox JSON export of label data.
        annotations_output_dir (str): File path of directory to write Pascal VOC
            annotation files.
        images_output_dir (str): File path of directory to write images.
        label_format (str): Format of the labeled data.
            Valid options are: "WKT" and "XY", default is "WKT".

    Todo:
        * Add functionality to allow use of local copy of an image instead of
            downloading it each time.
    """

    # make sure annotation output directory is valid
    try:
        annotations_output_dir = os.path.abspath(annotations_output_dir)
        assert os.path.isdir(annotations_output_dir)
    except AssertionError as e:
        logging.exception('Annotation output directory does not exist')
        return None

    # read labelbox JSON output
    with open(labeled_data) as f:
        label_data = json.loads(f.read())

    if use_local:
        image_id = 'External ID'
    else:
        image_id = 'ID'

    label_set = dict()
    for data in tqdm(label_data):
        labels = []
        if label_format == 'object':
            if 'objects' in data['Label']:
                for label in data['Label']['objects']:
                    labels.append(label['value'])
                if data[image_id] in label_set:
                    pass
                else:
                    label_set[data[image_id]] = labels

        try:
            write_label(
                label_id=data[image_id],
                image_url=os.path.join(local_image_dir, data['External ID']) if use_local else data['Labeled Data'],
                labels=data['Label'],
                label_format=label_format,
                images_output_dir=images_output_dir,
                annotations_output_dir=annotations_output_dir,
                database=database,
                local_image_dir=local_image_dir,
                use_local=use_local
            )

        except requests.exceptions.MissingSchema as e:
            logging.exception(('"Labeled Data" field must be a URL. Support for local files coming soon'))
            continue
        except requests.exceptions.ConnectionError as e:
            logging.exception(f"Failed to fetch image from {data['Labeled Data']}")
            continue
    write_image_set(label_set, image_sets_dir, use_local=use_local)


def write_image_set(label_set, image_set_dir=None, use_local=False):
    unique_labels = list(set([x[0] for x in label_set.values()]))

    if use_local:
        file_ending = ''
    else:
        file_ending = '.jpeg'

    try:
        for label in unique_labels:
            with open(os.path.join(image_set_dir, f"{label}.txt"), 'w+') as f:
                pass

            with open(os.path.join(image_set_dir, f"{label}.txt"), 'a+') as f:
                for image in label_set:
                    labels = label_set[image]
                    if label in labels:
                        f.write(f"{image}{file_ending} {1}\n")
                    else:
                        f.write(f"{image}{file_ending} {-1}\n")
    except TypeError as e:
        logging.exception(f"Please provide image sets directory, usually is PascalVOC-export-LB/ImageSets/Main'")


def write_label(label_id,
                image_url,
                labels,
                label_format,
                images_output_dir,
                annotations_output_dir,
                database='Unknown',
                use_local=False,
                local_image_dir=None):
    "Writes a Pascal VOC formatted image and label pair to disk."
    label_id = label_id.split('.')[0]
    # Download image and save it
    if use_local:
        im = Image.open(image_url)
    else:
        response = requests.get(image_url, stream=True)
        response.raw.decode_content = True
        im = Image.open(response.raw)

    image_name = (f'{label_id}.{im.format.lower()}')
    image_fqn = os.path.join(images_output_dir, image_name)
    im.save(image_fqn, format=im.format)

    # generate image annotation in Pascal VOC
    width, height = im.size
    xml_writer = PascalWriter(database=database, path=image_fqn, width=width, height=height)

    # remove classification labels (Skip, etc...)
    if not callable(getattr(labels, 'keys', None)):
        # skip if no categories (e.g. "Skip")
        return

    # convert label to Pascal VOC format
    for category_name, wkt_data in labels.items():
        if label_format == 'WKT':
            xml_writer = _add_pascal_object_from_wkt(
                xml_writer,
                img_height=height,
                wkt_data=wkt_data,
                label=category_name)

        elif label_format == 'XY':
            xml_writer = _add_pascal_object_from_xy(
                xml_writer,
                img_height=height,
                polygons=wkt_data,
                label=category_name)

        elif label_format == 'object':
            xml_writer = _add_pascal_object_from_bbox(
                xml_writer=xml_writer,
                img_height=height,
                img_width=width,
                bbox=wkt_data,
                label=category_name)
        else:
            e = UnknownFormatError(label_format=label_format)
            logging.exception(e.message)
            raise e

    # write Pascal VOC xml annotation for image
    xml_writer.save(os.path.join(annotations_output_dir, '{}.xml'.format(label_id)))


def _add_pascal_object_from_wkt(xml_writer, img_height, wkt_data, label):
    polygons = []
    if type(wkt_data) is list:  # V3+
        polygons = map(lambda x: wkt.loads(x['geometry']), wkt_data)
    else:  # V2
        polygons = wkt.loads(wkt_data)

    for m in polygons:
        xy_coords = []
        for x, y in m.exterior.coords:
            xy_coords.extend([x, img_height - y])
        # remove last polygon if it is identical to first point
        if xy_coords[-2:] == xy_coords[:2]:
            xy_coords = xy_coords[:-2]
        xml_writer.addObject(name=label, xy_coords=xy_coords)
    return xml_writer


def _add_pascal_object_from_xy(xml_writer, img_height, polygons, label):
    for polygon in polygons:
        if 'geometry' in polygon:  # V3
            polygon = polygon['geometry']
        assert type(polygon) is list  # V2 and V3

        xy_coords = []
        for x, y in [(p['x'], p['y']) for p in polygon]:
            xy_coords.extend([x, img_height - y])
        xml_writer.addObject(name=label, xy_coords=xy_coords)
    return xml_writer


def _add_pascal_object_from_bbox(xml_writer, img_height, img_width, bbox, label):
    new = True
    for obj in bbox:
        if 'bbox' in obj:  # V3
            bbox = obj['bbox']
        xy_coords = []
        xy_coords.extend([bbox['left'], bbox['top'], bbox['width'] + bbox['left'], bbox['height'] + bbox['top']])
        xml_writer.addObject(name=obj['value'], xy_coords=xy_coords, new=True)
    return xml_writer