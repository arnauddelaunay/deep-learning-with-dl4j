import os
from os.path import join as ospj
import glob
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

DATA_PATH = "data/images"


def load_data(root, vfold_ratio=0.2, max_items_per_class=4000):
    all_files = glob.glob(os.path.join(root, '*.npy'))

    # initialize variables
    x = np.empty([0, 784])
    y = np.empty([0])
    class_names = []

    # load each data file
    for idx, file in enumerate(all_files):
        data = np.load(file)
        data = data[0: max_items_per_class, :]
        labels = np.full(data.shape[0], idx)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)

        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)

    print("%d classes found : %s" % (len(class_names), class_names))
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    vfold_size = int(x.shape[0] / 100 * (vfold_ratio * 100))
    x_test = x[0: vfold_size, :]
    y_test = y[0: vfold_size]

    x_train = x[vfold_size: x.shape[0], :]
    y_train = y[vfold_size: y.shape[0]]
    return x_train, y_train, x_test, y_test, class_names


def dump_images(train_x, train_y, test_x, test_y, classes):
    if not os.path.isdir(DATA_PATH):
        os.makedirs(DATA_PATH)
    for folder, dataset, targets in [('train', train_x, train_y), ('test', test_x, test_y)]:
        for idx, img in enumerate(dataset):
            class_name = classes[int(targets[idx])]
            if not os.path.isdir(ospj(DATA_PATH, folder, class_name)):
                os.makedirs(ospj(DATA_PATH, folder, class_name))
            gray_img = img.reshape(28, 28)
            matplotlib.image.imsave(
                ospj(DATA_PATH, folder, class_name, "%s_%d.png" % (class_name, idx)),
                gray_img,
                cmap=plt.get_cmap('gray_r')
            )


if __name__ == "__main__":
    print("Loading data from npy and reshaping...")
    x_train, y_train, x_test, y_test, class_names = load_data('data')
    print("Dumping %d train images and %d test images in %s..." % (len(x_train), len(x_test), DATA_PATH))
    dump_images(x_train, y_train, x_test, y_test, class_names)
