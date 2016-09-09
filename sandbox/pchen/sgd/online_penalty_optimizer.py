from rllab.misc.ext import compile_function, lazydict, flatten_tensor_variables
from rllab.misc import logger
from rllab.core.serializable import Serializable
import theano.tensor as TT
import theano
import numpy as np
import scipy.optimize


class OnlinePenaltyOptimizer(Serializable):

    def __init__(
            self,
            optimizer,
            max_opt_itr=20,
            initial_penalty=1.0,
            min_penalty=1e-2,
            max_penalty=1e6,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            adapt_penalty=True,
            adapt_itr=32,
            data_split=None,
    ):
        Serializable.quick_init(self, locals())
        self._optimizer = optimizer
        self._max_opt_itr = max_opt_itr
        self._penalty = initial_penalty
        self._initial_penalty = initial_penalty
        self._min_penalty = min_penalty
        self._max_penalty = max_penalty
        self._increase_penalty_factor = increase_penalty_factor
        self._decrease_penalty_factor = decrease_penalty_factor
        self._adapt_penalty = adapt_penalty

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._min_constraint_val = None
        self._constraint_name = None
        self._adapt_itr = adapt_itr
        self._data_split = data_split

    def update_opt(self, loss, target, leq_constraint, inputs, constraint_name="constraint", *args, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """
        constraint_term, constraint_value = leq_constraint
        penalty_var = theano.shared(self._initial_penalty, 'penalty_coeff') # TT.scalar("penalty")
        penalized_loss = loss + penalty_var * constraint_term #+ \
                         # -TT.log(constraint_value*1.3 - constraint_term)
                        #penalty_var*(constraint_term > constraint_value)*(constraint_value - constraint_term)**2

        self._target = target
        self._max_constraint_val = constraint_value
        self._min_constraint_val = 0.8*constraint_value
        self._constraint_name = constraint_name
        self._penalty_var = penalty_var

        def get_opt_output():
            flat_grad = flatten_tensor_variables(theano.grad(
                penalized_loss, target.get_params(trainable=True), disconnected_inputs='ignore'
            ))
            return [penalized_loss.astype('float64'), flat_grad.astype('float64')]

        self._opt_fun = lazydict(
            f_loss=lambda: compile_function(inputs, loss, log_name="f_loss"),
            f_constraint=lambda: compile_function(inputs, constraint_term, log_name="f_constraint"),
            f_penalized_loss=lambda: compile_function(
                inputs=inputs, # + [penalty_var],
                outputs=[penalized_loss, loss, constraint_term],
                log_name="f_penalized_loss",
            ),
            f_opt=lambda: compile_function(
                inputs=inputs, # + [penalty_var],
                outputs=get_opt_output(),
                log_name="f_opt"
            )
        )

        self._optimizer.update_opt(penalized_loss, target, inputs, *args, **kwargs)

    def loss(self, inputs):
        return self._opt_fun["f_loss"](*inputs)

    def constraint_val(self, inputs):
        return self._opt_fun["f_constraint"](*inputs)

    def optimize(self, inputs):

        inputs = tuple(inputs)
        if self._data_split is not None:
            maxlen = len(inputs[0])
            cutoff = int(maxlen * self._data_split)
            val_inputs = tuple(
                inp[cutoff:] for inp in inputs
            )
            inputs = tuple(
                inp[:cutoff] for inp in inputs
            )
        f_penalized_loss = self._opt_fun["f_penalized_loss"]

        try_penalty = np.clip(
            self._penalty_var.get_value(), self._min_penalty, self._max_penalty)

        train = []
        val = []
        _, try_loss, try_constraint_val = f_penalized_loss(*inputs)
        _, val_loss, val_constraint_val = f_penalized_loss(*val_inputs)
        train.append((try_loss, try_constraint_val))
        val.append((val_loss, val_constraint_val))
        logger.log('before optim penalty %f => loss %f (%f), %s %f (%f)' %
                   (try_penalty, try_loss, val_loss,
                    self._constraint_name, try_constraint_val, val_constraint_val))
        for _ in self._optimizer.optimize_gen(inputs, yield_itr=self._adapt_itr):
            # logger.log('trying penalty=%.3f...' % try_penalty)

            # _, try_loss, try_constraint_val = f_penalized_loss(*inputs)

            # logger.log('penalty %f => loss %f, %s %f' %
            #            (try_penalty, try_loss, self._constraint_name, try_constraint_val))
            if self._data_split is not None:
                _, try_loss, try_constraint_val = f_penalized_loss(*inputs)
                _, val_loss, val_constraint_val = f_penalized_loss(*val_inputs)
                train.append((try_loss, try_constraint_val))
                val.append((val_loss, val_constraint_val))
                logger.log('penalty %f => loss %f (%f), %s %f (%f)' %
                           (try_penalty, try_loss, val_loss,
                            self._constraint_name, try_constraint_val, val_constraint_val))

            if not self._adapt_penalty:
                continue

            # Increase penalty if constraint violated, or if constraint term is NAN
            if try_constraint_val > self._max_constraint_val or np.isnan(try_constraint_val):
                penalty_scale_factor = self._increase_penalty_factor
            elif try_constraint_val <= self._min_constraint_val:
                # if constraint is lower than threshold, shrink penalty
                penalty_scale_factor = self._decrease_penalty_factor
            else:
                # if things are good, keep current penalty
                penalty_scale_factor = 1.
            try_penalty *= penalty_scale_factor
            try_penalty = np.clip(try_penalty, self._min_penalty, self._max_penalty)
            self._penalty_var.set_value(try_penalty)

        import matplotlib.pyplot as plt;
        plt.plot(range(len(train)), [v[0] for v in train]);
        plt.plot(range(len(train)), [v[0] for v in val]);
        # plt.show()
        import ipdb; ipdb.set_trace()

