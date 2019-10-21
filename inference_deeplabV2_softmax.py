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

# dir_img = "/home/luca/data/train_val_test_png/train_png/"

# LAB
dir_img = "Y:/tesisti/rossi/data/train_val_test_png/test_png/"


images = glob.glob(dir_img + "*.png")
images.sort()

print("load model weights")
for k in tqdm(range(len(images))):

    img = cv2.imread("C:/Users/Utente/Desktop/gatto.jpg")

    w, h, _ = img.shape
    scale = img / 127.5 - 1
    res = deeplab_model.predict(np.expand_dims(scale, 0))
    labels = res.squeeze()

    name = images[k]
    name = name[-15:]
    name = name[:11] + ".npy"

    pathTmp = "C:/Users/Utente/Desktop/" + name
    np.save(pathTmp, labels)
    cv2.waitKey()


