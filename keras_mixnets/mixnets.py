# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for MixNet model.
[1] Mingxing Tan, Quoc V. Le
  MixNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""
from typing import List

import math
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.models import Model
from keras.utils import get_file
from keras.utils import get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import preprocess_input as _preprocess
from keras_mixnets.config import BlockArgs
from keras_mixnets.config import get_mixnet_small, get_mixnet_medium, get_mixnet_large
from keras_mixnets.custom_objects import DropConnect
from keras_mixnets.custom_objects import MixNetConvInitializer
from keras_mixnets.custom_objects import MixNetDenseInitializer
from keras_mixnets.custom_objects import Swish
from keras_mixnets.custom_objects import GroupConvolution

__all__ = ['MixNet',
           'MixNetSmall',
           'MixNetMedium',
           'MixNetLarge',
           'preprocess_input']

GROUP_NUM = 1


def preprocess_input(x, data_format=None):
    return _preprocess(x, data_format, mode='torch', backend=K)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
def _split_channels(total_filters, num_groups):
    split = [total_filters // num_groups for _ in range(num_groups)]
    split[0] += total_filters - sum(split)
    return split


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_model.py
def round_filters(filters, depth_multiplier, depth_divisor, min_depth):
    """Round number of filters based on depth multiplier."""
    multiplier = float(depth_multiplier) if depth_multiplier is not None else None
    divisor = int(depth_divisor)
    min_depth = min_depth

    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_model.py
def round_repeats(repeats):
    """Round number of filters based on depth multiplier."""
    return int(repeats)


# Ontained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
class GroupedConv2D(object):
    """Groupped convolution.
    Currently tf.keras and tf.layers don't support group convolution, so here we
    use split/concat to implement this op. It reuses kernel_size for group
    definition, where len(kernel_size) is number of groups. Notably, it allows
    different group has different kernel_size size.
    """

    def __init__(self, filters, kernel_size, **kwargs):
        """Initialize the layer.
        Args:
          filters: Integer, the dimensionality of the output space.
          kernel_size: An integer or a list. If it is a single integer, then it is
            same as the original Conv2D. If it is a list, then we split the channels
            and perform different kernel_size for each group.
          **kwargs: other parameters passed to the original conv2d layer.
        """

        global GROUP_NUM
        self._groups = len(kernel_size)
        self._channel_axis = -1
        self.filters = filters
        self.kernels = kernel_size

        self._conv_kwargs = {
            'strides': kwargs.get('strides', (1, 1)),
            'dilation_rate': kwargs.get('dilation_rate', (1, 1)),
            'kernel_initializer': kwargs.get('kernel_initializer', MixNetConvInitializer()),
            'padding': 'same',
            'use_bias': kwargs.get('use_bias', False),
        }

        GROUP_NUM += 1

    def __call__(self, inputs):
        grouped_op = GroupConvolution(self.filters, self.kernels, groups=self._groups,
                                      type='conv', conv_kwargs=self._conv_kwargs)
        x = grouped_op(inputs)
        return x


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_model.py
def SEBlock(input_filters, se_ratio, expand_ratio, activation_fn, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()

    num_reduced_filters = max(
        1, int(input_filters * se_ratio))
    filters = input_filters * expand_ratio

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        x = layers.Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = GroupedConv2D(
            num_reduced_filters,
            kernel_size=[1],
            strides=[1, 1],
            kernel_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=True)(x)

        x = activation_fn()(x)

        # Excite
        x = GroupedConv2D(
            filters,
            kernel_size=[1],
            strides=[1, 1],
            kernel_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = layers.Activation('sigmoid')(x)
        out = layers.Multiply()([x, inputs])
        return out

    return block


# Obtained from
class MDConv(object):
    """MDConv with mixed depthwise convolutional kernels.
    MDConv is an improved depthwise convolution that mixes multiple kernels (e.g.
    3x3, 5x5, etc). Right now, we use an naive implementation that split channels
    into multiple groups and perform different kernels for each group.
    See Mixnet paper for more details.
    """

    def __init__(self, kernel_size, strides, dilated=False, **kwargs):
        """Initialize the layer.
        Most of args are the same as tf.keras.layers.DepthwiseConv2D except it has
        an extra parameter "dilated" to indicate whether to use dilated conv to
        simulate large kernel_size size. If dilated=True, then dilation_rate is ignored.
        Args:
          kernel_size: An integer or a list. If it is a single integer, then it is
            same as the original tf.keras.layers.DepthwiseConv2D. If it is a list,
            then we split the channels and perform different kernel_size for each group.
          strides: An integer or tuple/list of 2 integers, specifying the strides of
            the convolution along the height and width.
          dilated: Bool. indicate whether to use dilated conv to simulate large
            kernel_size size.
          **kwargs: other parameters passed to the original depthwise_conv layer.
        """
        self._channel_axis = -1
        self._dilated = dilated
        self.kernels = kernel_size

        self._conv_kwargs = {
            'strides': strides,
            'dilation_rate': kwargs.get('dilation_rate', (1, 1)),
            'kernel_initializer': kwargs.get('kernel_initializer', MixNetConvInitializer()),
            'padding': 'same',
            'use_bias': kwargs.get('use_bias', False),
        }

    def __call__(self, inputs):
        filters = K.int_shape(inputs)[self._channel_axis]
        grouped_op = GroupConvolution(filters, self.kernels, groups=len(self.kernels),
                                      type='depthwise_conv', conv_kwargs=self._conv_kwargs)
        x = grouped_op(inputs)
        return x


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_model.py
def MixNetBlock(input_filters, output_filters,
                dw_kernel_size, expand_kernel_size,
                project_kernel_size, strides,
                expand_ratio, se_ratio,
                id_skip, drop_connect_rate,
                batch_norm_momentum=0.99,
                batch_norm_epsilon=1e-3,
                swish=False,
                dilated=None,
                data_format=None):

    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_ratio
    relu_activation = Swish if swish else layers.ReLU

    def block(inputs):
        # Expand part
        if expand_ratio != 1:
            x = GroupedConv2D(
                filters,
                kernel_size=expand_kernel_size,
                strides=[1, 1],
                kernel_initializer=MixNetConvInitializer(),
                padding='same',
                use_bias=False)(inputs)

            x = layers.BatchNormalization(
                axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon)(x)

            x = relu_activation()(x)
        else:
            x = inputs

        kernel_size = dw_kernel_size
        # Depthwise Convolutional Phase
        x = MDConv(
            kernel_size,
            strides=strides,
            dilated=dilated,
            depthwise_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)
        x = relu_activation()(x)

        if has_se:
            x = SEBlock(input_filters, se_ratio, expand_ratio,
                        relu_activation,
                        data_format)(x)

        # output phase
        x = GroupedConv2D(
            output_filters,
            kernel_size=project_kernel_size,
            strides=[1, 1],
            kernel_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)

        if id_skip:
            if all(s == 1 for s in strides) and (
                    input_filters == output_filters):

                # only apply drop_connect if skip presents.
                # if drop_connect_rate:
                #     x = DropConnect(drop_connect_rate)(x)

                x = layers.Add()([x, inputs])

        return x

    return block


def MixNet(input_shape,
           block_args_list: List[BlockArgs],
           depth_multiplier: float,
           include_top=True,
           weights=None,
           input_tensor=None,
           pooling=None,
           classes=1000,
           dropout_rate=0.,
           drop_connect_rate=0.,
           batch_norm_momentum=0.99,
           batch_norm_epsilon=1e-3,
           depth_divisor=8,
           stem_size=16,
           feature_size=1536,
           min_depth=None,
           data_format=None,
           default_size=None,
           **kwargs):
    """
    Builder model for MixNets.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        block_args_list: Optional List of BlockArgs, each
            of which detail the arguments of the MixNetBlock.
            If left as None, it defaults to the blocks
            from the paper.
        depth_multiplier: Determines the number of channels
            available per layer. Compound Coefficient that
            needs to be found using grid search on a base
            configuration model.
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        batch_norm_momentum: Float, default batch normalization
            momentum. Obtained from the paper.
        batch_norm_epsilon: Float, default batch normalization
            epsilon. Obtained from the paper.
        depth_divisor: Optional. Used when rounding off the coefficient
             scaled channels and depth of the layers.
        min_depth: Optional. Minimum depth value in order to
            avoid blocks with 0 layers.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.
        default_size: Specifies the default image size of the model

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')

    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    if default_size is None:
        default_size = 224

    if block_args_list is None:
        block_args_list = get_mixnet_small()

    # count number of strides to compute min size
    stride_count = 1
    for block_args in block_args_list:
        if block_args.strides is not None and block_args.strides[0] > 1:
            stride_count += 1

    min_size = int(2 ** stride_count)

    # Determine proper input shape and default size.
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=min_size,
                                      data_format=data_format,
                                      require_flatten=include_top,
                                      weights=weights)

    # Stem part
    if input_tensor is None:
        inputs = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            inputs = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor

    # Stem part
    x = inputs
    x = GroupedConv2D(
        filters=round_filters(stem_size, depth_multiplier,
                              depth_divisor, min_depth),
        kernel_size=[3],
        strides=[2, 2],
        kernel_initializer=MixNetConvInitializer(),
        padding='same',
        use_bias=False)(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = layers.ReLU()(x)

    num_blocks = sum([block_args.num_repeat for block_args in block_args_list])
    drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

    # Blocks part
    for block_idx, block_args in enumerate(block_args_list):
        assert block_args.num_repeat > 0

        # Update block input and output filters based on depth multiplier.
        block_args.input_filters = round_filters(block_args.input_filters, depth_multiplier, depth_divisor, min_depth)
        block_args.output_filters = round_filters(block_args.output_filters, depth_multiplier, depth_divisor, min_depth)
        block_args.num_repeat = round_repeats(block_args.num_repeat)

        # The first block needs to take care of stride and filter size increase.
        x = MixNetBlock(block_args.input_filters, block_args.output_filters,
                        block_args.dw_kernel_size, block_args.expand_kernel_size,
                        block_args.project_kernel_size, block_args.strides,
                        block_args.expand_ratio, block_args.se_ratio,
                        block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                        batch_norm_momentum, batch_norm_epsilon, block_args.swish,
                        block_args.dilated, data_format)(x)

        if block_args.num_repeat > 1:
            block_args.input_filters = block_args.output_filters
            block_args.strides = [1, 1]

        for _ in range(block_args.num_repeat - 1):
            x = MixNetBlock(block_args.input_filters, block_args.output_filters,
                            block_args.dw_kernel_size, block_args.expand_kernel_size,
                            block_args.project_kernel_size, block_args.strides,
                            block_args.expand_ratio, block_args.se_ratio,
                            block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                            batch_norm_momentum, batch_norm_epsilon, block_args.swish,
                            block_args.dilated, data_format)(x)

    # Head part
    x = GroupedConv2D(
        filters=feature_size,
        kernel_size=[1],
        strides=[1, 1],
        kernel_initializer=MixNetConvInitializer(),
        padding='same',
        use_bias=False)(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = layers.ReLU()(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(data_format=data_format)(x)

        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

        x = layers.Dense(classes, kernel_initializer=MixNetDenseInitializer())(x)
        x = layers.Activation('softmax')(x)

    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    outputs = x

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)

    model = Model(inputs, outputs)

    return model


def MixNetSmall(input_shape=None,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                classes=1000,
                dropout_rate=0.2,
                drop_connect_rate=0.,
                data_format=None):
    """
    Builds MixNet B0.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return MixNet(input_shape,
                  get_mixnet_small(),
                  depth_multiplier=1.0,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes,
                  dropout_rate=dropout_rate,
                  drop_connect_rate=drop_connect_rate,
                  data_format=data_format,
                  default_size=224)


def MixNetMedium(input_shape=None,
                 include_top=True,
                 weights=None,
                 input_tensor=None,
                 pooling=None,
                 classes=1000,
                 dropout_rate=0.25,
                 drop_connect_rate=0.,
                 data_format=None):
    """
    Builds MixNet B1.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return MixNet(input_shape,
                  get_mixnet_medium(),
                  depth_multiplier=1.0,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes,
                  dropout_rate=dropout_rate,
                  drop_connect_rate=drop_connect_rate,
                  stem_size=24,
                  data_format=data_format,
                  default_size=224)


def MixNetLarge(input_shape=None,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                classes=1000,
                dropout_rate=0.3,
                drop_connect_rate=0.,
                data_format=None):
    """
    Builds MixNet B2.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return MixNet(input_shape,
                  get_mixnet_large(),
                  depth_multiplier=1.3,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes,
                  dropout_rate=dropout_rate,
                  drop_connect_rate=drop_connect_rate,
                  stem_size=24,
                  data_format=data_format,
                  default_size=224)


if __name__ == '__main__':
    import os
    from keras.models import load_model
    from keras.callbacks import TensorBoard

    model = MixNetSmall(include_top=True, weights=False)
    model.summary()

    # x = tf.zeros([10, 224, 224, 3])
    # out = model(x)

    # model.compile('adam', 'categorical_crossentropy')
    #
    # import numpy as np
    # data = np.zeros([10, 224, 224, 3])
    # labels = np.ones([10, 1000])
    #
    # model.fit(data, labels)

    import numpy as np
    params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("Param count : ", params)

    # tb = TensorBoard('./logs/')
    # tb.set_model(model)
    #
    # tb.on_epoch_end(0)

    # model.save("temp.h5")
    #
    # if os.path.exists('temp.h5'):
    #     model = load_model('temp.h5', compile=False)
    #     model.summary()
    #
    # else:
    #     raise FileNotFoundError("Keras model file not found !")
    #
    # if os.path.exists('temp.h5'):
    #     os.remove('temp.h5')
