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
"""Image warping using per-pixel flow vectors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def resize3D(img, shape, method=tf.image.ResizeMethod.BILINEAR):
    b_size, x_size, y_size, z_size, c_size = img.shape#.as_list()
    x_size_new, y_size_new, z_size_new = shape
    squeeze_b_x = tf.reshape(img, [-1, y_size, z_size, c_size])
    resize_b_x = tf.image.resize(squeeze_b_x, [y_size_new, z_size_new], method=tf.image.ResizeMethod.BILINEAR)
    resume_b_x = tf.reshape(resize_b_x, [b_size, x_size, y_size_new, z_size_new, c_size])
    reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
    squeeze_b_z = tf.reshape(reoriented, [-1, y_size_new, x_size, c_size])
    resize_b_z = tf.image.resize(squeeze_b_z, [y_size_new, x_size_new], method=tf.image.ResizeMethod.BILINEAR)
    resume_b_z = tf.reshape(resize_b_z, [b_size, z_size_new, y_size_new, x_size_new, c_size])
    output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
    return output_tensor

