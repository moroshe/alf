# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from absl import logging
from absl.testing import parameterized
import tensorflow as tf

from alf.utils.data_buffer import DataBuffer


class DataBufferTest(unittest.TestCase):
    def assertArrayEqual(self, x, y):
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(float(tf.reduce_max(abs(x - y))), 0)

    def test_data_buffer(self):
        dim = 20
        capacity = 256
        data_spec = [
            tf.TensorSpec(shape=(dim // 3, ), dtype=tf.float32),
            tf.TensorSpec(shape=(dim - dim // 3, ), dtype=tf.float32)
        ]

        data_buffer = DataBuffer(data_spec=data_spec, capacity=capacity)

        def _get_batch(batch_size):
            x = tf.random.normal(shape=(batch_size, dim))
            x = [x[..., :dim // 3], x[..., dim // 3:]]
            return x

        data_buffer.add_batch(_get_batch(100))
        self.assertEqual(int(data_buffer.current_size), 100)
        batch = _get_batch(1000)
        data_buffer.add_batch(batch)
        self.assertEqual(int(data_buffer.current_size), capacity)
        ret = data_buffer.get_batch_by_indices(
            (data_buffer._current_pos + tf.range(-capacity, 0)) % capacity)
        self.assertArrayEqual(ret[0], batch[0][-capacity:])
        self.assertArrayEqual(ret[1], batch[1][-capacity:])
        batch = _get_batch(100)
        data_buffer.add_batch(batch)
        ret = data_buffer.get_batch_by_indices(
            (data_buffer._current_pos + tf.range(-100, 0)) % capacity)
        self.assertArrayEqual(ret[0], batch[0])
        self.assertArrayEqual(ret[1], batch[1])


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()
    unittest.main()