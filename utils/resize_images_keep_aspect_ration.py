import uuid

from PIL import Image
from os import listdir
from os.path import isfile, join

PATH = 'IMAGE PATH'
NEW_PATH = 'NEW IMAGE PATH'

onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]

size = (416, 416)
keep_aspect_ratio = False
for img in onlyfiles:
    # Hele pathen til bildet, "joiner" PATH oppe med bildenavn
    path = join(PATH, img)

    # Sjekker om filen er gyldig
    if isfile(path):

        # Leser inn bildet, eventuelt kan man bruke:
        # image = cv2.imread(path)
        image = Image.open(path)

        # Her kommer eventuelt deres kode inn hvor dere endrer lysforhold i bildet, erstatt det under med
        # kode for å endre lysforhold
        if keep_aspect_ratio:
            new_img = image.thumbnail(size, Image.ANTIALIAS)
        else:
            new_img = image.resize(size)

        # Nytt navn til bildet, for eksempel kan uuid library brukes som følgende:
        new_name = str(uuid.uuid4())

        # deretter:
        new_path = join(NEW_PATH, new_name)
        new_img.save(new_path, "JPEG", optimize=True)
