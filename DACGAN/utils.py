import os
import glob
import cv2
import numpy as np
import pickle
import os.path
from tensorflow.keras.datasets.cifar10 import load_data


def load_dataset(image_size, create_new, cifar10):
    c1 = os.path.isfile("saved_dataset/X_train.pickle")
    c2 = os.path.isfile("saved_dataset/y_train.pickle")
    if cifar10:
        (X, y), (_, _) = load_data()
        # Normalize and format the data:
        X = np.array(X)
        X = (X - 127.5) / 127.5

        y = np.array(y)
        y = y.reshape(-1, 1)
        number_of_classes = 10
    else:
        if c1 and c2 and not create_new:

            with open('saved_dataset/X_train.pickle', 'rb') as data:
                X = pickle.load(data)

            with open('saved_dataset/y_train.pickle', 'rb') as data:
                y = pickle.load(data)

            number_of_classes = max(y)[0] + 1
            print("Dataset loaded successfully")
        else:
            X, y, number_of_classes = create_dataset(image_size)

    return X, y, number_of_classes


def create_dataset(image_size):

    label_ref = "Mapping from folder class to numberic label used: \n\n"
    X = []
    y = []
    folder_list = glob.glob("data/*")
    number_of_classes = len([folder for folder in folder_list if os.path.isdir(folder)])
    # Gå gjennom hver mappe.
    folder_counter = -1
    for folder_name in folder_list:
        if os.path.isdir(folder_name):
            folder_counter += 1
            label_ref += folder_name + " : " + str(folder_counter) + "\n"

            image_list = glob.glob(folder_name + "/*")
            # Gå gjennom hvert bilde i nåværende mappe
            for image_name in image_list:
                X, y = add_image(X, y, image_name, image_size, folder_counter)

    # Normaliser og formater data:
    X = np.array(X)
    X = (X - 127.5) / 127.5
    y = np.array(y)
    y = y.reshape(-1, 1)

    # Shuffle:
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    with open('saved_dataset/X_train.pickle', 'wb') as output:
        pickle.dump(X, output)

    with open('saved_dataset/y_train.pickle', 'wb') as output:
        pickle.dump(y, output)

    with open("labels.txt", "w") as text_file:
        print(label_ref, file=text_file)

    print("""Dataset correctly created and saved. 
              Total number of samples: """ + str(X.shape[0]))

    return X, y, number_of_classes


def add_image(X, y, image_name, image_size, folder_counter):
    img = cv2.imread(image_name)
    if img is not None:
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        X.append(img.astype(np.float32))
        y.append(np.uint8(folder_counter))

    else:
        print("Could not load ", image_name, "Is it an image?")

    return X, y
