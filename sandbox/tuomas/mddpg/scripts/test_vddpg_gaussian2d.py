from rllab.envs.base import Env
from rllab import spaces
import tensorflow as tf
import numpy as np


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Base learning rate.')
flags.DEFINE_integer('modes', 2, 'Number of modes.')
flags.DEFINE_boolean('fixed', False, 'Fixed target distribution.')


class MultimodalGaussianEnv(Env):

    def __init__(self, n_modes, action_lb, action_ub, fixed_observations):
        self._n_modes = n_modes
        self._action_lb = action_lb
        self._action_ub = action_ub
        self._fixed_observations = fixed_observations

    def step(self, action):
        return self.observe(), 0, True, {}

    def observe(self):
        return np.random.randn(1)

        #if self._fixed_observations is not None:
        #    n_cases = self._fixed_observations.shape[0]
        #    i = np.random.randint(low=0, high=n_cases)
        #    return self._fixed_observations[i]

        #modes = self._n_modes

        #weights = np.random.rand(modes)
        #weights = weights / weights.sum()
        #mus = np.random.randn(modes)*2 - 1
        #sigmas = np.random.rand(modes) + 0.5

        #obs = np.stack((weights, mus, sigmas), axis=1) # one row per mode
        #obs = np.ndarray.flatten(obs) # (w_1, mu_1, sigma_1, w_2, ...)

        #return obs

    def reset(self):
        return self.observe()

    @property
    def horizon(self):
        pass

    @property
    def action_space(self):
        return spaces.Box(self._action_lb, self._action_ub)

    @property
    def observation_space(self):
        #dummy = np.array([0] * self._n_modes * 3)
        dummy = np.array([0])
        return spaces.Box(dummy, dummy)


from sandbox.haoran.mddpg.qfunctions.nn_qfunction import NNCritic
from rllab.core.serializable import Serializable
class MixtureGaussian2DCritic(NNCritic):
    """ Q(s,a) is a 2D mixture of Gaussian in a """
    def __init__(
            self,
            scope_name,
            observation_dim,
            action_input,
            observation_input,
            reuse=False,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        super(MixtureGaussian2DCritic, self).__init__(
            scope_name=scope_name,
            observation_dim=observation_dim,
            action_dim=2,
            action_input=action_input,
            observation_input=observation_input,
            reuse=reuse,
            **kwargs
        )

    def create_network(self, action_input, observation_input):
        # obs = tf.unstack(observation_input, axis=1)
        weights = [1./2., 2./3.]
        mus = [np.array((-2., 0.)), np.array((1., 1.))]
        sigmas = [0.2, 1.0]
        a = action_input

        # TODO: assume all samples in a batch are the same (use index 0).
        # unnormalized density
        components = []
        for w, mu, sigma in zip(weights, mus, sigmas):
            mu = np.reshape(mu, (1, 2))

            comp = (1./tf.sqrt(2. * np.pi * sigma**2)) * tf.exp(
                -0.5 / sigma**2 * tf.reduce_sum(tf.square(a - mu), axis=1))

            components.append(comp)

        temp = 0.5
        output = temp * tf.log(tf.add_n(components))
        return output

    def get_weight_tied_copy(self, action_input, observation_input):
        """
        HT: basically, re-run __init__ with specified kwargs. In particular,
        the variable scope doesn't change, and self.observations_placeholder
        and NN params are reused.
        """
        return self.__class__(
            scope_name=self.scope_name,
            observation_dim=self.observation_dim,
            action_input=action_input,
            observation_input=observation_input,
            reuse=True,
        )

from sandbox.tuomas.mddpg.policies.stochastic_policy import \
    DummyExplorationStrategy

from sandbox.tuomas.mddpg.algos.vddpg import VDDPG
from rllab.misc.overrides import overrides
import matplotlib.pyplot as plt
class VDDPGTest(VDDPG):

    def eval_critic(self, o):
        xx = np.arange(-4, 4, 0.05)
        X, Y = np.meshgrid(xx, xx)
        all_actions = np.vstack([X.ravel(), Y.ravel()]).transpose()
        obs = np.array([o] * all_actions.shape[0])

        feed = {
            self.qf.observations_placeholder: obs,
            self.qf.actions_placeholder: all_actions
        }
        Q = self.sess.run(self.qf.output, feed).reshape(X.shape)
        return X, Y, Q

    @overrides
    def evaluate(self, epoch, es_path_returns):
        obs_single = self.pool.random_batch(1)['observations'].reshape((-1))
        X, Y, Q = self.eval_critic(obs_single)

        plt.clf()
        plt.contour(X, Y, Q, 20)
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))

        # sample and plot actions
        all_actions = []
        N = 250
        all_obs = np.array([obs_single] * N)
        all_actions = self.policy.get_actions(all_obs)[0]
        #for i in range(N):
        #    action, _ = self.policy.get_action(obs_single)
        #    all_actions.append(action.squeeze())

        for k, action in enumerate(all_actions):
            x = action[0]
            y = action[1]
            plt.plot(x, y, '*')
            #ax_qf.text(x, y, '%d'%(k))

        plt.draw()
        plt.pause(0.001)


# -------------------------------------------------------------------
def test():
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from sandbox.haoran.mddpg.kernels.gaussian_kernel import \
        SimpleAdaptiveDiagonalGaussianKernel, \
        SimpleDiagonalConstructor, DiagonalGaussianKernel
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
    from sandbox.tuomas.mddpg.policies.stochastic_policy \
        import StochasticNNPolicy

    K = 100 # number of particles

    adaptive_kernel = True

    ddpg_kwargs = dict(
        epoch_length=10,  # evaluate / plot per SVGD step
        min_pool_size=2,  # must be at least 2
        replay_pool_size=2,  # should only sample from recent experiences,
                        # note that the sample drawn can be one iteration older
        eval_samples=1,  # doesn't matter since we override evaluate()
        n_epochs=100000,  # number of SVGD steps
        policy_learning_rate=FLAGS.learning_rate,  # note: this is higher than DDPG's 1e-4
        qf_learning_rate=0.1,
        batch_size=32,  # only need recent samples, though it's slow
        alpha=1,  # 1 is the SVGD default
        train_actor=True,
        train_critic=False,
    )
    q_target_type = "none"  # do not update the critic

    # ----------------------------------------------------------------------

    env = TfEnv(MultimodalGaussianEnv(
        n_modes=FLAGS.modes,
        action_lb=np.array([-10, 10]),  # bounds on the particles
        action_ub=np.array([10, 10]),
        fixed_observations=True
    ))

    es = DummyExplorationStrategy()
    qf = MixtureGaussian2DCritic(
        "critic",
        observation_dim=env.observation_space.flat_dim,
        action_input=None,
        observation_input=None,
    )

    policy = StochasticNNPolicy(
        'actor',
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        hidden_dims=(128, 128),
        sample_dim=2,
        freeze_samples=False
    )

    if adaptive_kernel:
        kernel = SimpleAdaptiveDiagonalGaussianKernel(
            "kernel",
            dim=env.action_space.flat_dim,
            h_min=0,
        )
    else:
        diag_constructor = SimpleDiagonalConstructor(
            dim=env.action_space.flat_dim,
            sigma=0.1,
        )
        kernel = DiagonalGaussianKernel(
            "kernel",
            diag=diag_constructor.diag(),
        )
    algorithm = VDDPGTest(
        env=env,
        exploration_strategy=es,
        policy=policy,
        kernel=kernel,
        qf=qf,
        K=K,
        q_target_type=q_target_type,
        **ddpg_kwargs
    )
    algorithm.train()

if __name__ == "__main__":
    test() # this way variables defined in test() are not global, hence
    # not accessible to other methods, avoiding bugs
