import lasagne.layers as L
import lasagne.nonlinearities as NL

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.core.serializable import Serializable

from sandbox.carlos_snn.distributions.categorical import Categorical

from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.policies.base import StochasticPolicy
from rllab.spaces import Discrete


class CategoricalMLPPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=NL.tanh,
            prob_network=None,
    ):
        """
        :param env_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param prob_network: manually specified network for this policy, other network params
        are ignored
        :return:
        """
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)

        if prob_network is None:
            prob_network = MLP(
                input_shape=(env_spec.observation_space.flat_dim,),
                output_dim=env_spec.action_space.n,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=NL.softmax,
            )

        self._l_prob = prob_network.output_layer
        self._l_obs = prob_network.input_layer
        self._f_prob = ext.compile_function([prob_network.input_layer.input_var], L.get_output(
            prob_network.output_layer))

        self._dist = Categorical(env_spec.action_space.n)

        super(CategoricalMLPPolicy, self).__init__(env_spec)
        LasagnePowered.__init__(self, [prob_network.output_layer])

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None):
        return dict(prob=L.get_output(self._l_prob, {self._l_obs: obs_var}))

    @overrides
    def dist_info(self, obs, state_infos=None):
        return dict(prob=self._f_prob(obs))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        dist_info = dict((k, val[0]) for k, val in self.dist_info([flat_obs]).iteritems()) # removing extra dim
        action = self._dist.sample(dist_info)
        print "Hier observes {}, acts {} with info {}".format(observation, action, dist_info)
        return action, dist_info

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        dist_infos = self.dist_info(flat_obs)
        actions = map(self._dist.sample, dist_infos)
        return actions, dist_infos

    @property
    def distribution(self):
        return self._dist