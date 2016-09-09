import os
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.rein.algos.trpo_vime import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools
from rllab import config

stub(globals())

# Param ranges
seeds = list(range(10))
etas = [0.001, 0.01, 0.1, 1.0]
replay_pools = [True]
kl_ratios = [True]
normalize_rewards = [True]
reverse_kl_regs = [True]
n_itr_updates = [5]
kl_batch_sizes = [5]
use_kl_ratio_qs = [False]
stochastic_outputs = [False]
param_cart_product = itertools.product(
    stochastic_outputs, use_kl_ratio_qs, kl_batch_sizes, normalize_rewards, n_itr_updates, reverse_kl_regs, kl_ratios, replay_pools, etas, seeds
)

for stochastic_output, use_kl_ratio_q, kl_batch_size, normalize_reward, n_itr_update, reverse_kl_reg, kl_ratio, replay_pool, eta, seed in param_cart_product:

    mdp_class = DoublePendulumEnv
    mdp = NormalizedEnv(env=mdp_class())

    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(32,),
    )

    baseline = GaussianMLPBaseline(
        mdp.spec,
        regressor_args=dict(hidden_sizes=(32,)),
    )

    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=100,
        whole_paths=True,
        max_path_length=100,
        n_itr=10000,
        step_size=0.01,
        eta=eta,
        eta_discount=0.998,
        snn_n_samples=10,
        subsample_factor=1.0,
        use_reverse_kl_reg=reverse_kl_reg,
        use_replay_pool=replay_pool,
        use_kl_ratio=kl_ratio,
        n_itr_update=n_itr_update,
        normalize_reward=normalize_reward,
        kl_batch_size=kl_batch_size,
        use_kl_ratio_q=use_kl_ratio_q,
        stochastic_output=stochastic_output
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix=config.EXP_PREFIX + "_" + "doublependulum",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="lab_kube",
        dry=False,
    )
