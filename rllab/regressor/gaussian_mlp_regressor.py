from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.lasagne_layers import ParamLayer
from rllab.core.serializable import Serializable
from rllab.core.network import MLP
from rllab.misc import normal_dist
from rllab.misc.ext import compile_function
from rllab.optimizer.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.optimizer.lbfgs_optimizer import LbfgsOptimizer
from rllab.misc import logger
import theano.tensor as TT
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne

NONE = list()


class GaussianMLPRegressor(LasagnePowered, Serializable):

    """
    A class for performing regression by fitting a Gaussian distribution to the outputs.
    """

    def __init__(
            self,
            input_shape,
            output_dim,
            hidden_sizes=(32, 32),
            nonlinearity=NL.rectify,
            optimizer=None,
            use_trust_region=True,
            step_size=0.01,
            learn_std=True,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=None,
            # We can't use None here since None is actually a valid value!
            std_nonlinearity=NONE,
            name=None,
    ):
        """
        :param input_shape: Shape of the input data.
        :param output_dim: Dimension of output.
        :param hidden_sizes: Number of hidden units of each layer of the mean network.
        :param nonlinearity: Non-linearity used for each layer of the mean network.
        :param optimizer: Optimizer for minimizing the negative log-likelihood.
        :param use_trust_region: Whether to use trust region constraint.
        :param step_size: KL divergence constraint for each iteration
        :param learn_std: Whether to learn the standard deviations. Only effective if adaptive_std is False. If
        adaptive_std is True, this parameter is ignored, and the weights for the std network are always learned.
        :param adaptive_std: Whether to make the std a function of the states.
        :param std_share_network: Whether to use the same network as the mean.
        :param std_hidden_sizes: Number of hidden units of each layer of the std network. Only used if
        `std_share_network` is False. It defaults to the same architecture as the mean.
        :param std_nonlinearity: Non-linearity used for each layer of the std network. Only used if `std_share_network`
        is False. It defaults to the same non-linearity as the mean.
        """
        Serializable.quick_init(self, locals())

        if optimizer is None:
            if use_trust_region:
                optimizer = PenaltyLbfgsOptimizer()
            else:
                optimizer = LbfgsOptimizer()

        self._optimizer = optimizer

        mean_network = MLP(
            input_shape=input_shape,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            nonlinearity=nonlinearity,
            output_nl=None,
        )

        l_mean = mean_network.l_out

        if adaptive_std:
            if std_hidden_sizes is None:
                std_hidden_sizes = hidden_sizes
            if std_nonlinearity is NONE:
                std_nonlinearity = nonlinearity
            l_log_std = MLP(
                input_shape=input_shape,
                input_var=mean_network.l_in,
                output_dim=output_dim,
                hidden_sizes=std_hidden_sizes,
                nonlinearity=std_nonlinearity,
                output_nl=None,
            )
        else:
            l_log_std = ParamLayer(
                mean_network.l_in,
                num_units=output_dim,
                param=lasagne.init.Constant(0.),
                name="output_log_std",
                trainable=learn_std,
            )

        LasagnePowered.__init__(self, [l_mean, l_log_std])

        xs_var = mean_network.input_var
        ys_var = TT.matrix("ys")
        old_means_var = TT.matrix("old_means")
        old_log_stds_var = TT.matrix("old_log_stds")

        means_var = L.get_output(l_mean)
        log_stds_var = L.get_output(l_log_std)

        mean_kl = TT.mean(normal_dist.kl_sym(
            old_means_var, old_log_stds_var, means_var, log_stds_var))

        loss = - TT.mean(normal_dist.log_likelihood_sym(ys_var, means_var, log_stds_var))

        self._f_predict = compile_function([xs_var], means_var)
        self._f_pdists = compile_function([xs_var], [means_var, log_stds_var])

        optimizer_args = dict(
            loss=loss,
            target=self,
        )

        if use_trust_region:
            optimizer_args["leq_constraint"] = (mean_kl, step_size)
            optimizer_args["inputs"] = [xs_var, ys_var, old_means_var, old_log_stds_var]
        else:
            optimizer_args["inputs"] = [xs_var, ys_var]

        self._optimizer.update_opt(**optimizer_args)

        self._use_trust_region = use_trust_region
        self._name = name

    def fit(self, xs, ys):
        if self._use_trust_region:
            old_means, old_log_stds = self._f_pdists(xs)
            inputs = [xs, ys, old_means, old_log_stds]
        else:
            inputs = [xs, ys]
        loss_before = self._optimizer.loss(*inputs)
        if self._name:
            prefix = self._name + "_"
        else:
            prefix = ""
        logger.record_tabular(prefix + 'LossBefore', loss_before)
        self._optimizer.optimize(*inputs)
        loss_after = self._optimizer.loss(*inputs)
        logger.record_tabular(prefix + 'LossAfter', loss_after)
        logger.record_tabular(prefix + 'dLoss', loss_before - loss_after)

    def predict(self, xs):
        return self._f_predict(xs)

    def predict_log_likelihood(self, xs, ys):
        means, log_stds = self._f_pdists(xs)
        return normal_dist.log_likelihood(ys, means, log_stds)

    def get_param_values(self, **tags):
        return LasagnePowered.get_param_values(self, **tags)

    def set_param_values(self, flattened_params, **tags):
        return LasagnePowered.set_param_values(self, flattened_params, **tags)