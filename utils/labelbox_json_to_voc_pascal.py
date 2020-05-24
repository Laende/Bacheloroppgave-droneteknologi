import labelbox2pascal as lb2pa
import os
import json
from os.path import join

os.chdir('..')
DATA_DIR = 'DATADIR'

labeled_data = join(DATA_DIR, 'data/EXPORT.json')
labeled_data_no_duplicates = join(DATA_DIR, 'data/EXPORT-no-duplicates.json')

local_image_dir = 'LOKAL BILDEMAPPE'

ann_output_dir = join(DATA_DIR, 'dataset/PascalVOC-OG\Annotations')
images_output_dir = join(DATA_DIR, 'dataset/PascalVOC-OG/JPEGImages')
image_sets_dir = join(DATA_DIR, 'dataset/PascalVOC-OG/ImageSets/Main')

# Denne funksjonen fjerne duplikater fra eksportfilen om disse finnes, og lagrer en ny fil uten duplikater.
def remove_duplicates_from_label_box(export_file, new_export_file):
    with open(export_file) as f:
        temp_label_data = json.loads(f.read())

    with open(export_file, 'w') as outfile:
        json.dump(temp_label_data, outfile, sort_keys=True, indent=4)

    seen = set()
    new_label_data = []

    for dic in temp_label_data:
        key = (dic['DataRow ID'])
        if key in seen:
            continue
        new_label_data.append(dic)
        seen.add(key)

    with open(new_export_file, 'w') as outfile:
        json.dump(new_label_data, outfile, sort_keys=True, indent=4)


remove_duplicates_from_label_box(labeled_data, labeled_data_no_duplicates)

lb2pa.from_json(
    labeled_data=labeled_data_no_duplicates,
    annotations_output_dir=ann_output_dir,
    images_output_dir=images_output_dir,
    image_sets_dir=image_sets_dir,
    label_format='object',
    database='fulldata - version 6 - labelbox - reworked',
    use_local=False,
    local_image_dir=''
)