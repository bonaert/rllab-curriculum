


from rllab.algos.trpo import TRPO
from rllab.baselines.extreme_linear_baseline import ExtremeLinearBaseline
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize

import itertools

stub(globals())

# Name
exp_prefix = "linear-baseline-squared-noise-1"

# Settings
local = False
debug = False
visualize = False
n_itr = 100

# Experiment parameters
envs = [HalfCheetahEnv(), Walker2DEnv()]
gae_lambdas = [1]
discounts = [0.99]
step_sizes = [0.01]
seeds = [1, 21, 31, 41, 51]
batch_sizes = [1000, 4000, 10000]
lookaheads = [0, 1, 2, 5, 10, 100]
max_path_lengths = [100]

# Handling code for experiment configuration
envs = [normalize(env) for env in envs]
plot = local and visualize
mode = "local" if local else "ec2"
terminate_machine = not debug
if debug:
    exp_prefix += "-DEBUG"

# Experiments
configurations = list(itertools.product(
    envs,
    gae_lambdas,
    discounts,
    step_sizes,
    batch_sizes,
    seeds,
    max_path_lengths,
    lookaheads))
if debug:
    configurations = [configurations[0]]
    lookaheads = [10]
    n_itr = 5
if not local:
    print("Number of EC2 instances to launch: {}".format(len(configurations)))
for env, gae_lambda, discount, step_size, batch_size, seed, mpl, lookahead in configurations:

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )

    baseline = ExtremeLinearBaseline(env.spec, lookahead=lookahead, max_path_length=mpl)

    linear_baseline_algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=mpl,
        n_itr=n_itr,
        gae_lambda=gae_lambda,
        discount=discount,
        step_size=step_size,
        plot=plot,
    )

    run_experiment_lite(
        linear_baseline_algo.train(),
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        plot=plot,
        mode=mode,
        exp_prefix=exp_prefix,
        terminate_machine=terminate_machine
    )
