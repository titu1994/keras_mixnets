import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import initializers
from keras import layers
from keras import models
from keras.utils import conv_utils
from keras.utils.generic_utils import get_custom_objects


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
class MixNetConvInitializer(initializers.Initializer):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas base_path we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.

    # Arguments:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    # Returns:
      an initialization for the variable
    """
    def __init__(self):
        super(MixNetConvInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()

        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return tf.random_normal(
            shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
class MixNetDenseInitializer(initializers.Initializer):
    """Initialization for dense kernels.
        This initialization is equal to
          tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                          distribution='uniform').
        It is written out explicitly base_path for clarity.

        # Arguments:
          shape: shape of variable
          dtype: dtype of variable
          partition_info: unused

        # Returns:
          an initialization for the variable
    """
    def __init__(self):
        super(MixNetDenseInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()

        init_range = 1.0 / np.sqrt(shape[1])
        return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
class Swish(layers.Layer):

    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        return tf.nn.swish(inputs)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
class DropConnect(layers.Layer):

    def __init__(self, drop_connect_rate=0., **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.drop_connect_rate = float(drop_connect_rate)

    def call(self, inputs, training=None):

        def drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate

            # Compute drop_connect tensor
            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = (inputs / keep_prob) * binary_tensor
            return output

        return K.in_train_phase(drop_connect, inputs, training=training)

    def get_config(self):
        config = {
            'drop_connect_rate': self.drop_connect_rate,
        }
        base_config = super(DropConnect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GroupConvolution(models.Model):

    def __init__(self, filters, kernels, groups,
                 type='conv', conv_kwargs=None,
                 **kwargs):
        super(GroupConvolution, self).__init__(**kwargs)

        if conv_kwargs is None:
            conv_kwargs = {
                'strides': (1, 1),
                'padding': 'same',
                'dilation_rate': (1, 1),
                'use_bias': False,
            }

        self.filters = filters
        self.kernels = kernels
        self.groups = groups
        self.type = type
        self.strides = conv_kwargs.get('strides', (1, 1))
        self.padding = conv_kwargs.get('padding', 'same')
        self.dilation_rate = conv_kwargs.get('dilation_rate', (1, 1))
        self.use_bias = conv_kwargs.get('use_bias', False)
        self.conv_kwargs = conv_kwargs or {}

        assert type in ['conv', 'depthwise_conv']
        if type == 'conv':
            splits = self._split_channels(filters, self.groups)
            self._layers = [layers.Conv2D(splits[i], kernels[i],
                                          strides=self.strides,
                                          padding=self.padding,
                                          dilation_rate=self.dilation_rate,
                                          use_bias=self.use_bias,
                                          kernel_initializer=MixNetConvInitializer())
                            for i in range(groups)]

        else:
            self._layers = [layers.DepthwiseConv2D(kernels[i],
                                                   strides=self.strides,
                                                   padding=self.padding,
                                                   dilation_rate=self.dilation_rate,
                                                   use_bias=self.use_bias,
                                                   kernel_initializer=MixNetConvInitializer())
                            for i in range(groups)]

        self.data_format = 'channels_last'
        self._channel_axis = -1

    def call(self, inputs, **kwargs):
        if len(self._layers) == 1:
            return self._layers[0](inputs)

        filters = K.int_shape(inputs)[self._channel_axis]
        splits = self._split_channels(filters, self.groups)
        x_splits = tf.split(inputs, splits, self._channel_axis)
        x_outputs = [c(x) for x, c in zip(x_splits, self._layers)]
        x = layers.concatenate(x_outputs, axis=self._channel_axis)
        return x

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                filter_size=1,
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.filters,)

    def _split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernels': self.kernels,
            'groups': self.groups,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'type': self.type,
            'conv_kwargs': self.conv_kwargs,
        }
        base_config = super(GroupConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({
    'MixNetConvInitializer': MixNetConvInitializer,
    'MixNetDenseInitializer': MixNetDenseInitializer,
    'DropConnect': DropConnect,
    'Swish': Swish,
    'GroupConvolution': GroupConvolution,
})

