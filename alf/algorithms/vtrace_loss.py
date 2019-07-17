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
"""Loss for vtrace algorithm."""

import gin
import tensorflow as tf

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.utils import common as tfa_common
from tf_agents.specs import tensor_spec

from alf.algorithms.rl_algorithm import TrainingInfo
from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.algorithms.actor_critic_loss import _normalize_advantages
from alf.utils.losses import element_wise_squared_loss
from alf.utils import common


@gin.configurable
#comment
class VTraceLoss(ActorCriticLoss):
    def __init__(self,
                 action_spec,
                 gamma=0.99,
                 td_error_loss_fn=element_wise_squared_loss,
                 td_lambda=0.95,
                 use_td_lambda_return=True,
                 normalize_advantages=True,
                 advantage_clip=None,
                 entropy_regularization=None,
                 td_loss_weight=1.0,
                 check_numerics=False,
                 debug_summaries=False):
        """Create a VTraceLoss object

        Implement V-Trace corrected loss in Section (4.2) of "IMPALA:
        Scalable Distributed Deep-RL with Importance Weighted Actor-Learner
        Architectures" https://arxiv.org/abs/1802.01561

        The total loss is decomposed into three components:
        policy_gradient_loss
         + td_loss_weight * td_loss (value function error)
         - entropy_regularization * entropy

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            gamma (float): A discount factor for future rewards.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            td_lambda (float): Lambda parameter for TD-lambda computation.
            normalize_advantages (bool): If True, normalize advantage to zero
                mean and unit variance within batch for caculating policy
                gradient.
            advantage_clip (float): If set, clip advantages to [-x, x]
            entropy_regularization (float): Coefficient for entropy
                regularization loss term.
            td_loss_weight (float): the weigt for the loss of td error.
            check_numerics (bool):  If true, adds tf.debugging.check_numerics to
                help find NaN / Inf values. For debugging only.

        """

        super(VTraceLoss, self).__init__(
            action_spec=action_spec,
            gamma=gamma,
            td_error_loss_fn=td_error_loss_fn,
            use_gae=False,
            td_lambda=td_lambda,
            use_td_lambda_return=use_td_lambda_return,
            normalize_advantages=normalize_advantages,
            advantage_clip=advantage_clip,
            entropy_regularization=entropy_regularization,
            td_loss_weight=td_loss_weight,
            debug_summaries=debug_summaries)

        self._check_numerics = check_numerics

    def _calc_returns_and_advantages(self, training_info, value):
        returns = value_ops.discounted_return(
            rewards=training_info.reward,
            values=value,
            step_types=training_info.step_type,
            discounts=training_info.discount * self._gamma)
        returns = common.tensor_extend(returns, value[-1])

        if not self._use_gae:
            advantages = returns - value
        else:
            advantages = value_ops.generalized_advantage_estimation(
                rewards=training_info.reward,
                values=value,
                step_types=training_info.step_type,
                discounts=training_info.discount * self._gamma,
                td_lambda=self._lambda)
            advantages = common.tensor_extend_zero(advantages)
            if self._use_td_lambda_return:
                returns = advantages + value

        return returns, advantages
