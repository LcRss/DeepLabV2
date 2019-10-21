import cv2
import glob
from DeeplabV2_resnet101 import ResNet101
from myMetrics import *
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

print("load model weights")
deeplab_model = ResNet101(input_shape=(None, None, 3), classes=21)
path = "Y:/tesisti/rossi/data/weights_resnet_deeplab_pascal_72_73mIoU/checkpoint_72_28mIoU_20k_321_batch3.hdf5"
# path = "Y:/tesisti/rossi/data/weights_resnet_deeplab_mscoco/deeplabV2_resnet101_byname.h5"
deeplab_model.load_weights(path, by_name=True)


# dir_img = "/home/luca/data/dataset_test/test_png/"
# dir_img = "/home/luca/data/datasetPascal/images_prepped_test/"

# dir_seg = '/home/luca/data/gray/test/'
# dir_seg = '/home/luca/data/gray/val/'

# dir_img = "/home/luca/data/train_val_test_png/val_png/"
# dir_seg = "/home/luca/data/segmentation_gray/gray/val/"

# LAB
dir_img = "Y:/tesisti/rossi/data/train_val_test_png/val_png/"
dir_seg = "Y:/tesisti/rossi/data/segmentation_gray/gray/val/"
####

images = glob.glob(dir_img + "*.png")
images.sort()
segs = glob.glob(dir_seg + "*.png")
segs.sort()

mapLabel = [
    (0, 0, 0),
    (0, 0, 128),
    (0, 128, 0),
    (0, 128, 128),
    (128, 0, 0),
    (128, 0, 128),
    (128, 128, 0),
    (128, 128, 128),
    (0, 0, 64),
    (0, 0, 192),
    (0, 128, 64),
    (0, 128, 192),
    (128, 0, 64),
    (128, 0, 192),
    (128, 128, 64),
    (128, 128, 192),
    (0, 64, 0),
    (0, 64, 128),
    (0, 192, 0),
    (0, 192, 128),
    (128, 64, 0),
]

# mat = tf.keras.backend.zeros(shape=(21, 21), dtype="int32")
mat = np.zeros(shape=(21, 21), dtype=np.int32)
print("load model weights")
for k in tqdm(range(len(images))):
    # for k in tqdm( range(0, 2)):
    # print(images[k])
    # print(k)

    img = cv2.imread(images[k])
    seg = cv2.imread(segs[k], cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("i", img)
    # cv2.waitKey()

    # h_z, w_z = seg.shape
    # segNew = np.zeros((h_z, w_z, 3), np.uint8)
    # for i in range(0, 21):
    #     mask = cv2.inRange(seg, i, i)
    #     v = mapLabel[i]
    #     segNew[mask > 0] = v

    # cv2.imshow("s", segNew)
    # cv2.waitKey()

    w, h, _ = img.shape
    scale = img / 127.5 - 1
    res = deeplab_model.predict(np.expand_dims(scale, 0))
    labels = np.argmax(res.squeeze(), -1)
    z = labels.astype(np.uint8)

    # cv2.imshow("pred",z)
    # cv2.waitKey()

    # y_pred = tf.keras.backend.reshape(z, [-1])
    # y_true = tf.keras.backend.reshape(seg, [-1])

    tmp = confusion_matrix(seg.flatten(), z.flatten(),
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

    # tmp = tf.math.confusion_matrix(labels=y_true, predictions=y_pred, num_classes=21)
    mat = mat + tmp

    # TO color labels and save image

    # h_z, w_z = z.shape
    # imgNew = np.zeros((h_z, w_z, 3), np.uint8)
    # for i in range(0, 21):
    #     mask = cv2.inRange(z, i, i)
    #     v = mapLabel[i]
    #     imgNew[mask > 0] = v

    # name = images[k]
    # name = name[-15:]

    # cv2.imshow("c", imgNew)
    # cv2.waitKey()
    # pathTmp = "Y:/tesisti/rossi\data\segmentation_gray/results_deeplabV2/test/" + name
    # cv2.imwrite(pathTmp, z)

#
iou = compute_and_print_IoU_per_class(confusion_matrix=mat, num_classes=21)
print(iou)
