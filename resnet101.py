from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Input
from tensorflow.python.keras.utils.layer_utils import get_source_inputs
import os

BASE_WEIGHTS_PATH = (
    'https://github.com/keras-team/keras-applications/'
    'releases/download/resnet/')
WEIGHTS_HASHES = {
    'resnet101': ('f1aeb4b969a6efcfb50fad2f0c20cfc5',
                  '88cf7a10940856eca736dc7b7e228a21')
}


def ResNet(stack_fn,
           preact,
           use_bias,
           model_name='resnet',
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           pooling=None,
           classes=1000):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        preact: whether to use pre-activation or not
            (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    # if weights == 'imagenet' and include_top and classes != 1000:
    #     raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
    #                      ' as true, `classes` should be 1000')

    # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=32,
    #                                   data_format=backend.image_data_format(),
    #                                   require_flatten=include_top,
    #                                   weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    bn_axis = 3

    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1', padding='same')(img_input)

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)

    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = stack_fn(x)

    # DeeplabV2

    # hole = 6
    # b1 = ZeroPadding2D(padding=(6, 6))(x)
    b1 = Conv2D(filters=21, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu', name='fc1_voc12_c0',
                padding='same')(x)
    # b1 = Dropout(0.5)(b1)
    # b1 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_1')(b1)
    # b1 = Dropout(0.5)(b1)
    # b1 = Conv2D(filters=21, kernel_size=(1, 1), activation='relu', name='fc8_voc12_1')(b1)

    # hole = 12
    # b2 = ZeroPadding2D(padding=(12, 12))(x)
    b2 = Conv2D(filters=21, kernel_size=(3, 3), dilation_rate=(12, 12), activation='relu', name='fc1_voc12_c1',
                padding='same')(x)
    # b2 = Dropout(0.5)(b2)
    # b2 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_2')(b2)
    # b2 = Dropout(0.5)(b2)
    # b2 = Conv2D(filters=21, kernel_size=(1, 1), activation='relu', name='fc8_voc12_2')(b2)

    # hole = 18
    # b3 = ZeroPadding2D(padding=(18, 18))(x)
    b3 = Conv2D(filters=21, kernel_size=(3, 3), dilation_rate=(18, 18), activation='relu', name='fc1_voc12_c2',
                padding='same')(x)
    # b3 = Dropout(0.5)(b3)
    # b3 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_3')(b3)
    # b3 = Dropout(0.5)(b3)
    # b3 = Conv2D(filters=21, kernel_size=(1, 1), activation='relu', name='fc8_voc12_3')(b3)

    # hole = 24
    # b4 = ZeroPadding2D(padding=(24, 24))(x)
    b4 = Conv2D(filters=21, kernel_size=(3, 3), dilation_rate=(24, 24), activation='relu', name='fc1_voc12_c3',
                padding='same')(x)
    # b4 = Dropout(0.5)(b4)
    # b4 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_4')(b4)
    # b4 = Dropout(0.5)(b4)
    # b4 = Conv2D(filters=21, kernel_size=(1, 1), activation='relu', name='fc8_voc12_4')(b4)

    s = Add()([b1, b2, b3, b4])

    logits = Lambda(lambda xx: tf.image.resize_bilinear(xx,
                                                        size=tf.shape(img_input)[1:3]))(s)

    out = Activation('softmax')(logits)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, out, name=model_name)

    # # Load weights.
    # if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):
    #     if include_top:
    #         file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'
    #         file_hash = WEIGHTS_HASHES[model_name][0]
    #     else:
    #         file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
    #         file_hash = WEIGHTS_HASHES[model_name][1]
    #     weights_path = get_file(file_name,
    #                             BASE_WEIGHTS_PATH + file_name,
    #                             cache_subdir='models',
    #                             file_hash=file_hash)
    #     # load by name sbagliato
    #     model.load_weights(weights_path, by_name=True)
    # elif weights is not None:
    #     model.load_weights(weights)

    return model


def block1(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None):
    """A residual block.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3  # if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = Conv2D(4 * filters, 1, strides=stride, padding='same', use_bias=False,
                          name='res%s_branch1' % name)(x)
        shortcut = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='bn%s_branch1' % name)(shortcut)
    else:
        shortcut = x

    x = Conv2D(filters, 1, strides=stride, name='res%s_branch2a' % name, use_bias=False, padding='same')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn%s_branch2a' % name)(x)
    x = Activation('relu', name=name + '_1_relu')(x)
    #####
    x = Conv2D(filters, kernel_size, padding='same', use_bias=False,
               name='res%s_branch2b' % name)(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn%s_branch2b' % name)(x)
    x = Activation('relu', name=name + '_2_relu')(x)
    #####
    x = Conv2D(4 * filters, 1, name='res%s_branch2c' % name, padding='same', use_bias=False, )(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn%s_branch2c' % name)(x)
    #####
    x = Add(name='res%s' % name)([shortcut, x])
    x = Activation('relu', name='res%s_relu' % name)(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name)
    if name == '2a':
        x = block1(x, filters, conv_shortcut=False, name='2b')
        x = block1(x, filters, conv_shortcut=False, name='2c')
    elif name == '5a':
        x = block1(x, filters, conv_shortcut=False, name='5b')
        x = block1(x, filters, conv_shortcut=False, name='5c')
    elif name == '3a':
        for i in range(1, blocks):
            x = block1(x, filters, conv_shortcut=False, name='3b' + str(i))
    elif name == '4a':
        for i in range(1, blocks):
            x = block1(x, filters, conv_shortcut=False, name='4b' + str(i))

    return x


def ResNet101(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='2a')
        x = stack1(x, 128, 4, name='3a')
        x = stack1(x, 256, 23, name='4a')
        x = stack1(x, 512, 3, name='5a')

        return x

    return ResNet(stack_fn, False, True, 'resnet101',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes)
