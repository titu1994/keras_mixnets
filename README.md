# Keras MixNets: Mixed Depthwise Convolutional Kernels
Keras Implementation of MixNets from the paper [MixNets: : Mixed Depthwise Convolution Kernels](https://arxiv.org/abs/1907.09595).

Code ported from the official codebase [https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet](https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet)

# Mixed Depthwise Convolutional Kernel

<img src="https://github.com/titu1994/keras_mixnets/blob/master/images/MixedConv.png" height=100% width=100%>

From the above paper, a Mixed Convolution is a group of convolutions with varying filter sizes. The paper suggests that [3x3, 5x5, 7x7] can be used safely without any loss in performance (and possible increase in performance), while a 9x9 or 11x11 may degrade performance if used without proper architecture search.

# Installation

## From PyPI:

```$ pip install keras_mixnets```

## From Master branch:

```
pip install git+https://github.com/titu1994/keras_mixnets.git

OR

git clone https://github.com/titu1994/keras_mixnets.git
cd keras_mixnets
pip install .
```

# Usage

Due to the use of Model Subclassing, the keras model built **cannot* be deserialized using `load_model`. You must build the model each time. tf.keras supports writing Layers which have additional Layers within them, but as Keras itself does not support it yet, these models cannot be deserialized using `load_model`.

```python

from keras_mixnets import MixNetSmall  # Medium and Large can also be used

model = MixNetSmall((224, 224, 3), include_top=True)
```

# Weights

Weights for these models have not been ported yet from Tensorflow.

# Requirements

 - Tensorflow 1.14+ (Not 2.x)
 - Keras 2.2.4+
