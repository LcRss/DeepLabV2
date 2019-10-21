import cv2
import glob
from DeeplabV3_mobilenetv2.model import Deeplabv3
from myMetrics import *
from tqdm import tqdm

BACKBONE = 'mobilenetv2'
deeplab_model = Deeplabv3(weights='pascal_voc', backbone=BACKBONE, input_shape=(None, None, 3), classes=21, OS=8)

# ENTIRE DATASET
dir_img = "/home/luca/data/dataset_instance_class_part/image_png/"
####

images = glob.glob(dir_img + "*.png")
images.sort()

for k in tqdm(range(len(images))):
    # for k in tqdm( range(0, 10)):

    img = cv2.imread(images[k])
    w, h, _ = img.shape
    scale = img / 127.5 - 1

    res = deeplab_model.predict(np.expand_dims(scale, 0))
    labels = np.argmax(res.squeeze(), -1)
    z = labels.astype(np.uint8)

    name = images[k]
    name = name[-15:]

    # cv2.imshow("c",z)
    # cv2.waitKey()
    pathTmp = "/home/luca/data/output_results/result_dataset_os_8/" + name
    cv2.imwrite(pathTmp, z)

