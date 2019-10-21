import numpy as np
import cv2
import itertools
import glob
import random
import tensorflow as tf
from PIL import Image
import io
from tensorflow.python.keras.callbacks import LearningRateScheduler


# get_img_seg & data_loader give input data and label
def get_img_seg(path_img, path_seg, height, width, num_classes, resize):
    img = cv2.imread(path_img)
    # AGGIUNTA
    img = img / 127.5 - 1

    seg = cv2.imread(path_seg, cv2.IMREAD_GRAYSCALE)

    h = img.shape[0]
    w = img.shape[1]

    # each layer of this array is a mask for a specific object
    if resize:

        # seg_labels = np.zeros((height, width, num_classes))

        if h <= w:

            start = random.randint(0, w - h)

            img = img[0:h, start: start + h]
            img = cv2.resize(src=img, dsize=(height, width), interpolation=cv2.INTER_LINEAR)

            seg = seg[0:h, start: start + h]
            seg = cv2.resize(src=seg, dsize=(height, width), interpolation=cv2.INTER_NEAREST)

        else:

            start = random.randint(0, h - w)

            img = img[start:start + w, 0: w]
            img = cv2.resize(src=img, dsize=(height, width), interpolation=cv2.INTER_LINEAR)

            seg = seg[start:start + w, 0: w]
            seg = cv2.resize(src=seg, dsize=(height, width), interpolation=cv2.INTER_NEAREST)
    # else:
    #     seg_labels = np.zeros((h, w, num_classes))

    seg_labels = tf.keras.utils.to_categorical(y=seg, num_classes=num_classes, dtype='uint8')
    # seg_labels = tf.one_hot(seg, num_classes)

    # for c in range(num_classes):
    #     seg_labels[:, :, c] = (seg == c).astype(int)

    return img, seg_labels


def data_loader(dir_img, dir_seg, batch_size, h, w, num_classes, resize):
    # list of all image path png
    print(dir_img)
    images = glob.glob(dir_img + "*.png")
    images.sort()
    # list of all seg img path
    print(dir_seg)
    segmentations = glob.glob(dir_seg + "*.png")
    segmentations.sort()

    # create an iterator of tuples ( img and its seg_img)
    zipped = itertools.cycle(zip(images, segmentations))

    while 1:

        X = []
        Y = []

        for _ in range(batch_size):
            im_path, seg_path = next(zipped)
            i, s = get_img_seg(im_path, seg_path, h, w, num_classes, resize)
            X.append(i)
            Y.append(s)

        yield np.array(X), np.array(Y)


def data_loader_Val(dir_img, dir_seg, batch_size, h, w, num_classes, resize):
    print("Val")
    # list of all image path png
    images = glob.glob(dir_img + "*.png")
    images.sort()
    # list of all seg img path
    segmentations = glob.glob(dir_seg + "*.png")
    segmentations.sort()

    # create an iterator of tuples ( img and its seg_img)
    zipped = itertools.cycle(zip(images, segmentations))

    X = []
    Y = []

    for _ in range(batch_size):
        im_path, seg_path = next(zipped)
        i, s = get_img_seg(im_path, seg_path, h, w, num_classes, resize)
        X.append(i)
        Y.append(s)
    print("End Val load")
    return np.array(X), np.array(Y)


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """

    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


def print_var(num_classes, batch_sz, pathTr, pathTrSeg, pathVal, pathValSeg, h, w, tr_sz, val_sz):
    # Print var
    print('Variables')

    print('num classes: ' + str(num_classes))
    print('batch size: ' + str(batch_sz))
    print('img height: ' + str(h))
    print('img width: ' + str(w))
    print('path imgs train: ' + pathTr)
    print('path imgs train seg: ' + pathTrSeg)
    print('dt train size: ' + str(tr_sz))
    print('path imgs val: ' + pathVal)
    print('path imgs val seg: ' + pathValSeg)
    print('dt val size: ' + str(val_sz))

# def step_decay_schedule(initial_lr=1e-3, decay_factor=0.7, step_size=10):
#
#     def schedule(epoch):
#         vl = initial_lr * (decay_factor ** (epoch // step_size))
#         return vl
#
#     return LearningRateScheduler(schedule, verbose=1)