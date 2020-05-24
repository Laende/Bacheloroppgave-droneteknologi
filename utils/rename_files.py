import os
import uuid

PATH = 'FILE DIR'

if __name__ == "__main__":
    files = os.listdir(path=PATH)
    for index, file in enumerate(files):
        try:
            os.rename(os.path.join(PATH, file), os.path.join(PATH, ''.join([str(uuid.uuid4()), '.jpg'])))
        except FileExistsError as e:
            print(e)
