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
"""Make various external gin-configurable objects."""

import gin
import gin.tf.external_configurables
import gym
import tensorflow as tf
from tf_agents.specs.tensor_spec import BoundedTensorSpec
from tf_agents.networks.utils import mlp_layers
from tf_agents.networks.sequential_layer import SequentialLayer

from tf_agents.environments import atari_wrappers

from alf.utils import math_ops

tf.keras.layers.Conv2D = gin.external_configurable(tf.keras.layers.Conv2D,
                                                   'tf.keras.layers.Conv2D')
tf.optimizers.Adam = gin.external_configurable(tf.optimizers.Adam,
                                               'tf.optimizers.Adam')
gin.external_configurable(tf.keras.layers.Concatenate,
                          'tf.keras.layers.Concatenate')

# This allows the environment creation arguments to be configurable by supplying
# gym.envs.registration.EnvSpec.make.ARG_NAME=VALUE
gym.envs.registration.EnvSpec.make = gin.external_configurable(
    gym.envs.registration.EnvSpec.make, 'gym.envs.registration.EnvSpec.make')

# Activation functions.
gin.external_configurable(tf.math.exp, 'tf.math.exp')

gin.external_configurable(tf.TensorSpec, 'tf.TensorSpec')

gin.external_configurable(BoundedTensorSpec, 'tf.BoundedTensorSpec')

gin.external_configurable(SequentialLayer, 'SequentialLayer')

gin.external_configurable(mlp_layers, 'mlp_layers')

gin.external_configurable(tf.keras.activations.softsign,
                          'tf.keras.activations.softsign')

gin.external_configurable(tf.keras.initializers.GlorotUniform,
                          'tf.keras.initializers.GlorotUniform')
