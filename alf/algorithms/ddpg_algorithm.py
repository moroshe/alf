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
"""Deep Deterministic Policy Gradient (DDPG)."""

import gin.tf
import tensorflow as tf

import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.actor_rnn_network import ActorRnnNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.ddpg.critic_rnn_network import CriticRnnNetwork

from tf_agents.networks.network import Network
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.utils import common as tfa_common

from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import namedtuple
from alf.data_structures import ActionTimeStep, Experience, LossInfo, TrainingInfo
from alf.utils import losses, common

DdpgCriticState = namedtuple("DdpgCriticState",
                             ['critic', 'target_actor', 'target_critic'])
DdpgCriticInfo = namedtuple("DdpgCriticInfo", ["q_value", "target_q_value"])
DdpgActorState = namedtuple("DdpgActorState", ['actor', 'critic'])
DdpgState = namedtuple("DdpgState", ['actor', 'critic'])
DdpgInfo = namedtuple(
    "DdpgInfo", ["action_distribution", "actor_loss", "critic"],
    default_value=())
DdpgLossInfo = namedtuple('DdpgLossInfo', ('actor', 'critic'))


@gin.configurable
class DdpgAlgorithm(OffPolicyAlgorithm):
    """Deep Deterministic Policy Gradient (DDPG).

    Reference:
    Lillicrap et al "Continuous control with deep reinforcement learning"
    https://arxiv.org/abs/1509.02971
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 actor_network: Network,
                 critic_network: Network,
                 ou_stddev=0.2,
                 ou_damping=0.15,
                 critic_loss=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 gradient_clipping=None,
                 debug_summaries=False,
                 name="DdpgAlgorithm"):
        """Create a DdpgAlgorithm.

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            actor_network (Network):  The network will be called with
                call(observation, step_type).
            critic_network (Network): The network will be called with
                call(observation, action, step_type).
            ou_stddev (float): Standard deviation for the Ornstein-Uhlenbeck
                (OU) noise added in the default collect policy.
            ou_damping (float): Damping factor for the OU noise added in the
                default collect policy.
            critic_loss (None|OneStepTDLoss): an object for calculating critic
                loss. If None, a default OneStepTDLoss will be used.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between [-dqda_clipping, dqda_clipping].
                Does not perform clipping if dqda_clipping == 0.
            actor_optimizer (tf.optimizers.Optimizer): The optimizer for actor.
            critic_optimizer (tf.optimizers.Optimizer): The optimizer for actor.
            gradient_clipping (float): Norm length to clip gradients.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        train_state_spec = DdpgState(
            actor=DdpgActorState(
                actor=actor_network.state_spec,
                critic=critic_network.state_spec),
            critic=DdpgCriticState(
                critic=critic_network.state_spec,
                target_actor=actor_network.state_spec,
                target_critic=critic_network.state_spec))

        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=train_state_spec,
            optimizer=[actor_optimizer, critic_optimizer],
            trainable_module_sets=[[actor_network], [critic_network]],
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            name=name)

        self._actor_network = actor_network
        self._critic_network = critic_network
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer

        self._target_actor_network = actor_network.copy(
            name='target_actor_network')
        self._target_critic_network = critic_network.copy(
            name='target_critic_network')

        self._ou_stddev = ou_stddev
        self._ou_damping = ou_damping

        if critic_loss is None:
            critic_loss = OneStepTDLoss(debug_summaries=debug_summaries)
        self._critic_loss = critic_loss

        self._ou_process = create_ou_process(action_spec, ou_stddev,
                                             ou_damping)

        self._update_target = common.get_target_updater(
            models=[self._actor_network, self._critic_network],
            target_models=[
                self._target_actor_network, self._target_critic_network
            ],
            tau=target_update_tau,
            period=target_update_period)

        self._dqda_clipping = dqda_clipping

    def predict(self, time_step: ActionTimeStep, state, epsilon_greedy):
        action, state = self._actor_network(
            time_step.observation,
            step_type=time_step.step_type,
            network_state=state.actor.actor)
        empty_state = tf.nest.map_structure(lambda x: (),
                                            self.train_state_spec)

        def _sample(a, ou):
            return tf.cond(
                tf.less(tf.random.uniform((), 0, 1),
                        epsilon_greedy), lambda: a + ou(), lambda: a)

        noisy_action = tf.nest.map_structure(_sample, action, self._ou_process)
        noisy_action = tf.nest.map_structure(tfa_common.clip_to_spec,
                                             noisy_action, self._action_spec)
        state = empty_state._replace(
            actor=DdpgActorState(actor=state, critic=()))
        return PolicyStep(
            action=noisy_action,
            state=state,
            info=DdpgInfo(action_distribution=action))

    def rollout(self,
                time_step: ActionTimeStep,
                state=None,
                mode=RLAlgorithm.ROLLOUT):
        if self.need_full_rollout_state():
            raise NotImplementedError("Storing RNN state to replay buffer "
                                      "is not supported by DdpgAlgorithm")
        return self.predict(time_step, state, epsilon_greedy=1.0)

    def _critic_train_step(self, exp: Experience, state: DdpgCriticState):
        target_action, target_actor_state = self._target_actor_network(
            exp.observation,
            step_type=exp.step_type,
            network_state=state.target_actor)
        target_q_value, target_critic_state = self._target_critic_network(
            (exp.observation, target_action),
            step_type=exp.step_type,
            network_state=state.target_critic)

        q_value, critic_state = self._critic_network(
            (exp.observation, exp.action),
            step_type=exp.step_type,
            network_state=state.critic)

        state = DdpgCriticState(
            critic=critic_state,
            target_actor=target_actor_state,
            target_critic=target_critic_state)

        info = DdpgCriticInfo(q_value=q_value, target_q_value=target_q_value)

        return state, info

    def _actor_train_step(self, exp: Experience, state: DdpgActorState):
        action, actor_state = self._actor_network(
            exp.observation, exp.step_type, network_state=state.actor)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(action)
            q_value, critic_state = self._critic_network(
                (exp.observation, action), network_state=state.critic)

        dqda = tape.gradient(q_value, action)

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = tf.clip_by_value(dqda, -self._dqda_clipping,
                                        self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                tf.stop_gradient(dqda + action), action)
            loss = tf.reduce_sum(loss, axis=list(range(1, len(loss.shape))))
            return loss

        actor_loss = tf.nest.map_structure(actor_loss_fn, dqda, action)
        state = DdpgActorState(actor=actor_state, critic=critic_state)
        info = LossInfo(
            loss=tf.add_n(tf.nest.flatten(actor_loss)), extra=actor_loss)
        return PolicyStep(action=action, state=state, info=info)

    def train_step(self, exp: Experience, state: DdpgState):
        critic_state, critic_info = self._critic_train_step(
            exp=exp, state=state.critic)
        policy_step = self._actor_train_step(exp=exp, state=state.actor)
        return policy_step._replace(
            state=DdpgState(actor=policy_step.state, critic=critic_state),
            info=DdpgInfo(
                action_distribution=policy_step.action,
                critic=critic_info,
                actor_loss=policy_step.info))

    def calc_loss(self, training_info: TrainingInfo):
        critic_loss = self._critic_loss(
            training_info=training_info,
            value=training_info.info.critic.q_value,
            target_value=training_info.info.critic.target_q_value)

        actor_loss = training_info.info.actor_loss

        return LossInfo(
            loss=critic_loss.loss + actor_loss.loss,
            extra=DdpgLossInfo(
                critic=critic_loss.extra, actor=actor_loss.extra))

    def after_train(self, training_info):
        self._update_target()

    def _trainable_attributes_to_ignore(self):
        return ['_target_actor_network', '_target_critic_network']


def create_ou_process(action_spec, ou_stddev, ou_damping):
    """Create nested zero-mean Ornstein-Uhlenbeck processes.

    The temporal update equation is:
    `x_next = (1 - damping) * x + N(0, std_dev)`

    Args:
        action_spec (nested BountedTensorSpec): action spec
        ou_damping (float): Damping rate in the above equation. We must have
            0 <= damping <= 1.
        ou_stddev (float): Standard deviation of the Gaussian component.
    Returns:
        nested OUProcess with the same structure as action_spec.
    """
    # todo with seed None
    seed_stream = tfp.util.SeedStream(seed=None, salt='ou_noise')

    def _create_ou_process(action_spec):
        return tfa_common.OUProcess(
            lambda: tf.zeros(action_spec.shape, dtype=action_spec.dtype),
            ou_damping,
            ou_stddev,
            seed=seed_stream())

    ou_process = tf.nest.map_structure(_create_ou_process, action_spec)
    return ou_process
