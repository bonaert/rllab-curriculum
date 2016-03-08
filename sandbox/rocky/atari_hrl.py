import os
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.mdp.openai_atari_mdp import AtariMDP
from rllab.mdp.subgoal_mdp import SubgoalMDP
from rllab.policy.subgoal_policy import SubgoalPolicy
from rllab.policy.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baseline.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baseline.subgoal_baseline import SubgoalBaseline
from rllab.algo.ppo import PPO
from rllab.algo.batch_hrl import BatchHRL
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.state_given_goal_mi_evaluator import StateGivenGoalMIEvaluator
from sandbox.rocky.zero_bonus_evaluator import ZeroBonusEvaluator
import lasagne.nonlinearities as NL

stub(globals())


mdp = SubgoalMDP(
    AtariMDP(rom_name="pong", obs_type="ram", frame_skip=4),
    n_subgoals=1,
)

algo = BatchHRL(
    high_algo=PPO(
        discount=0.99,
        step_size=0.01,
        max_penalty_itr=3,
    ),
    low_algo=PPO(
        discount=0.99,
        step_size=0.01,
        max_penalty_itr=3,
    ),
    batch_size=10000,
    whole_paths=True,
    max_path_length=4500,
    n_itr=200,
    discount=0.99,
)

policy = SubgoalPolicy(
    mdp=mdp,
    high_policy=CategoricalMLPPolicy(
        mdp=mdp.high_mdp,
        hidden_sizes=[32, 32],
        nonlinearity=NL.rectify,
    ),
    low_policy=CategoricalMLPPolicy(
        mdp=mdp.low_mdp,
        hidden_sizes=[32, 32],
        nonlinearity=NL.rectify,
    ),
)

baseline = SubgoalBaseline(
    mdp=mdp,
    high_baseline=GaussianMLPBaseline(
        mdp=mdp.high_mdp,
        regressor_args=dict(
            hidden_sizes=[32, 32],
            nonlinearity=NL.rectify,
            step_size=0.06,
        ),
    ),
    low_baseline=GaussianMLPBaseline(
        mdp=mdp.low_mdp,
        regressor_args=dict(
            hidden_sizes=[32, 32],
            nonlinearity=NL.rectify,
            step_size=0.06,
        ),
    ),
)

# bonus_evaluator = ZeroBonusEvaluator()
bonus_evaluator = StateGivenGoalMIEvaluator(
    mdp=mdp,
    regressor_args=dict(
        hidden_sizes=[32, 32],
        nonlinearity=NL.rectify,
        step_size=0.06,
    ),
)

run_experiment_lite(
    algo.train(mdp=mdp, policy=policy, baseline=baseline, bonus_evaluator=bonus_evaluator),
    exp_prefix="ppo_atari",
    n_parallel=4,
    snapshot_mode="last",
    seed=1,
)