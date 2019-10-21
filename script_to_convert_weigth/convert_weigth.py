import os

import numpy as np
import tensorflow as tf
from resnet101 import ResNet101

filePath = "/home/luca/PycharmProjects/deeplab_113/deeplab_resnet.ckpt"


def get_filename(key):
    """Rename tensor name to the corresponding Keras layer weight name.
    # Arguments
        key: tensor name in TF (determined by tf.variable_scope)
    """
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('MobilenetV2_', '')
    filename = filename.replace('BatchNorm', 'BN')
    if 'Momentum' in filename:
        return None

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def extract_tensors_from_checkpoint_file(filename, output_folder='weights'):
    """Extract tensors from a TF checkpoint file.
    # Arguments
        filename: TF checkpoint file
        output_folder: where to save the output numpy array files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reader = tf.train.NewCheckpointReader(filename)
    keys = reader.get_variable_to_shape_map()
    f1 = open('./weights/testfile', 'w+')
    for key in keys:
        # convert tensor name into the corresponding Keras layer weight name and save

        filename = get_filename(key)
        if filename:
            path = os.path.join(output_folder, filename)
            arr = reader.get_tensor(key)
            np.save(path, arr)
            print(filename, file=f1)


if not os.path.exists("./weights/resnet_deeplab"):
    os.makedirs("./weights/resnet_deeplab")

extract_tensors_from_checkpoint_file(
    filePath, output_folder='./weights/resnet_deeplab')

print('Instantiating an empty model...')

model = ResNet101()

WEIGHTS_DIR = '/home/luca/PycharmProjects/deeplab_113/Segmentation_model/weights/resnet_deeplab/'

print('Loading weights from', WEIGHTS_DIR)
layer_model = model.layers

for layer in layer_model:
    if layer.weights:
        weights = []
        for w in layer.weights:

            weight_name = os.path.basename(w.name).replace(':0', '')
            weight_file = layer.name + '_' + weight_name + '.npy'
            weight_arr = np.load(os.path.join(WEIGHTS_DIR, weight_file))
            weights.append(weight_arr)

        layer.set_weights(weights)

print('Saving model weights...')
OUTPUT_WEIGHT_FILENAME = 'deeplabV2_resnet101_tf_dim_ordering_tf_kernels.h5'
if not os.path.exists("./weights/resnet_deeplab_model"):
    os.makedirs("./weights/resnet_deeplab_model")
model.save_weights(os.path.join("./weights/resnet_deeplab_model", OUTPUT_WEIGHT_FILENAME))


# lys = deeplab_model.layers
# f1 = open('./prova', 'w+')
# for layer in lys:
#     if layer.weights:
#         weights = []
#         for w in layer.weights:
#             print(w, file=f1)