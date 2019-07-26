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

import re


class BlockArgs(object):

    def __init__(self, input_filters=None,
                 output_filters=None,
                 dw_kernel_size=None,
                 expand_kernel_size=None,
                 project_kernel_size=None,
                 strides=None,
                 num_repeat=None,
                 se_ratio=None,
                 expand_ratio=None,
                 identity_skip=True,
                 swish=False,
                 dilated=False):

        self.input_filters = input_filters
        self.output_filters = output_filters
        self.dw_kernel_size = self._normalize_kernel_size(dw_kernel_size)
        self.expand_kernel_size = self._normalize_kernel_size(expand_kernel_size)
        self.project_kernel_size = self._normalize_kernel_size(project_kernel_size)
        self.strides = strides
        self.num_repeat = num_repeat
        self.se_ratio = se_ratio
        self.expand_ratio = expand_ratio
        self.identity_skip = identity_skip
        self.swish = swish
        self.dilated = dilated

    def decode_block_string(self, block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        self.input_filters = int(options['i'])
        self.output_filters = int(options['o'])
        self.dw_kernel_size = self._parse_ksize(options['k'])
        self.expand_kernel_size = self._parse_ksize(options['a'])
        self.project_kernel_size = self._parse_ksize(options['p'])
        self.num_repeat = int(options['r'])
        self.identity_skip = ('noskip' not in block_string)
        self.se_ratio = float(options['se']) if 'se' in options else None
        self.expand_ratio = int(options['e'])
        self.strides = [int(options['s'][0]), int(options['s'][1])]
        self.swish = 'sw' in block_string
        self.dilated = 'dilated' in block_string

        return self

    def encode_block_string(self, block):
        """Encodes a block to a string.

        Encoding Schema:
        "rX_kX_sXX_eX_iX_oX{_se0.XX}{_noskip}"
         - X is replaced by a any number ranging from 0-9
         - {} encapsulates optional arguments

        To deserialize an encoded block string, use
        the class method :
        ```python
        BlockArgs.from_block_string(block_string)
        ```
        """

        args = [
            'r%d' % block.num_repeat,
            'k%s' % self._encode_ksize(block.kernel_size),
            'a%s' % self._encode_ksize(block.expand_kernel_size),
            'p%s' % self._encode_ksize(block.project_kernel_size),
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]

        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)

        if block.id_skip is False:
            args.append('noskip')

        if block.swish:
            args.append('sw')

        if block.dilated:
            args.append('dilated')

        return '_'.join(args)

    def _normalize_kernel_size(self, val):
        if type(val) == int:
            return [val]

        return val

    def _parse_ksize(self, ss):
        return [int(k) for k in ss.split('.')]

    def _encode_ksize(self, arr):
        return '.'.join([str(k) for k in arr])

    @classmethod
    def from_block_string(cls, block_string):
        """
        Encoding Schema:
        "rX_kX_sXX_eX_iX_oX{_se0.XX}{_noskip}"
         - X is replaced by a any number ranging from 0-9
         - {} encapsulates optional arguments

        To deserialize an encoded block string, use
        the class method :
        ```python
        BlockArgs.from_block_string(block_string)
        ```

        Returns:
            BlockArgs object initialized with the block
            string args.
        """
        block = cls()
        return block.decode_block_string(block_string)


# Default list of blocks for MixNets
def get_mixnet_small(depth_multiplier=None):
    blocks_args = [
        'r1_k3_a1_p1_s11_e1_i16_o16',
        'r1_k3_a1.1_p1.1_s22_e6_i16_o24',
        'r1_k3_a1.1_p1.1_s11_e3_i24_o24',

        'r1_k3.5.7_a1_p1_s22_e6_i24_o40_se0.5_sw',
        'r3_k3.5_a1.1_p1.1_s11_e6_i40_o40_se0.5_sw',

        'r1_k3.5.7_a1_p1.1_s22_e6_i40_o80_se0.25_sw',
        'r2_k3.5_a1_p1.1_s11_e6_i80_o80_se0.25_sw',

        'r1_k3.5.7_a1.1_p1.1_s11_e6_i80_o120_se0.5_sw',
        'r2_k3.5.7.9_a1.1_p1.1_s11_e3_i120_o120_se0.5_sw',

        'r1_k3.5.7.9.11_a1_p1_s22_e6_i120_o200_se0.5_sw',
        'r2_k3.5.7.9_a1_p1.1_s11_e6_i200_o200_se0.5_sw',
    ]
    DEFAULT_BLOCK_LIST = [BlockArgs.from_block_string(s)
                          for s in blocks_args]

    return DEFAULT_BLOCK_LIST


def get_mixnet_medium(depth_multiplier=None):
    blocks_args = [
        'r1_k3_a1_p1_s11_e1_i24_o24',
        'r1_k3.5.7_a1.1_p1.1_s22_e6_i24_o32',
        'r1_k3_a1.1_p1.1_s11_e3_i32_o32',

        'r1_k3.5.7.9_a1_p1_s22_e6_i32_o40_se0.5_sw',
        'r3_k3.5_a1.1_p1.1_s11_e6_i40_o40_se0.5_sw',

        'r1_k3.5.7_a1_p1_s22_e6_i40_o80_se0.25_sw',
        'r3_k3.5.7.9_a1.1_p1.1_s11_e6_i80_o80_se0.25_sw',

        'r1_k3_a1_p1_s11_e6_i80_o120_se0.5_sw',
        'r3_k3.5.7.9_a1.1_p1.1_s11_e3_i120_o120_se0.5_sw',

        'r1_k3.5.7.9_a1_p1_s22_e6_i120_o200_se0.5_sw',
        'r3_k3.5.7.9_a1_p1.1_s11_e6_i200_o200_se0.5_sw',
    ]

    DEFAULT_BLOCK_LIST = [BlockArgs.from_block_string(s)
                          for s in blocks_args]

    return DEFAULT_BLOCK_LIST


def get_mixnet_large(depth_multiplier=None):
    return get_mixnet_medium(depth_multiplier)
