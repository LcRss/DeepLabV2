from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import argparse

from Utils import *
from MyCustomCallback import CallbackmIoU
import os

from DeeplabV2_resnet101 import ResNet101

parser = argparse.ArgumentParser()

parser.add_argument("--input_height", type=int, default=321)
parser.add_argument("--input_width", type=int, default=321)
parser.add_argument("--epochs", type=int, default=37)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--patience", type=int, default=10)

args = parser.parse_args()

h_img = args.input_height
w_img = args.input_width
epochs_sz = args.epochs
batch_sz = args.batch_size
lr_p = args.lr
pat = args.patience

num_cl = 21  # 20
train_sz = 4498
valid_sz = 500
rsize = True

prefix = "DeeplabV2_resnet101_preTrain_"+str(epochs_sz)+"_"
decay = 0
weights = None
activation = 'softmax'

# batch_sz = 12
# epochs_sz = 100
# num_cl = 21  # 20
# train_sz = 4498
# valid_sz = 500
# h_img = 256
# w_img = 256
# rsize = True
#
# prefix = "ResNet_Deeplab_preTrain_"
# lr_p = 0.0001
# decay = 0
# weights = None
# activation = 'softmax'

# Train
train_images_path = "Y:/tesisti/rossi/data/train_val_test_png/train_png/"
train_segs_path = "Y:/tesisti/rossi/data/segmentation_gray/gray/train/"
# Val
val_images_path = 'Y:/tesisti/rossi/data/train_val_test_png/val_png/'
val_segs_path = 'Y:/tesisti/rossi/data/segmentation_gray/gray/val/'

path = "./" + prefix + "Train_class_21_lr_" + str(lr_p) + "_batch_" + str(
    batch_sz) + "_size_" + str(h_img) + "/"
if not os.path.isdir(path):
    os.mkdir(path)

pathTBoard = "./" + path + "Graph_deeplab/"
if not os.path.isdir(pathTBoard):
    os.mkdir(pathTBoard)

pathTChPoints = "./" + path + "Checkpoints_deeplab/"
if not os.path.isdir(pathTChPoints):
    os.mkdir(pathTChPoints)

pathWeight = "./" + path + "Weight_deeplab/"
if not os.path.isdir(pathWeight):
    os.mkdir(pathWeight)

print('LOAD DATA')
G1 = data_loader(dir_img=train_images_path, dir_seg=train_segs_path, batch_size=batch_sz, h=h_img, w=w_img,
                 num_classes=num_cl, resize=rsize)
G2 = data_loader(dir_img=val_images_path, dir_seg=val_segs_path, batch_size=batch_sz, h=h_img, w=w_img,
                 num_classes=num_cl, resize=rsize)

print('DEEPLAB')
deeplab_model = ResNet101(input_shape=(None, None, 3), classes=21)

path = "Y:/tesisti/rossi/data/weights_resnet_deeplab_mscoco/deeplabV2_resnet101_tf_dim_ordering_tf_kernels.h5"
# path = "Y:/tesisti/rossi/data/weights_resnet_deeplab_mscoco/deeplabV2_resnet101_byname.h5"
deeplab_model.load_weights(path, by_name=True)
# deeplab_model.save_weights("Y:/tesisti/rossi/data/weights_resnet_deeplab_mscoco/deeplabV2_resnet101_byname.h5")

# print(deeplab_model.summary())

deeplab_model.compile(optimizer=optimizers.SGD(lr=lr_p, momentum=0.9, decay=0, nesterov=True),
                      loss=['categorical_crossentropy'],
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])

cb_tensorBoard = TensorBoard(log_dir=pathTBoard, histogram_freq=0, write_graph=True,
                             write_grads=False,
                             write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                             embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

cb_earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', patience=10, min_delta=0,
                                 restore_best_weights=True)

cb_modelCheckPoint = ModelCheckpoint(
    filepath=pathTChPoints + 'checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5',
    monitor='val_loss',
    save_best_only=False, save_weights_only=True, mode='auto', period=2)

print('FIT')

history = deeplab_model.fit_generator(generator=G1, steps_per_epoch=train_sz // batch_sz, epochs=epochs_sz, verbose=1,
                                      callbacks=[cb_tensorBoard, cb_earlyStopping, cb_modelCheckPoint,
                                                 CallbackmIoU(path, lr_p)],
                                      validation_data=G2,
                                      validation_steps=valid_sz // batch_sz)
