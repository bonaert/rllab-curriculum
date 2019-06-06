import numpy as np
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator

# These values are described in the hyper-parameters section of the papers

NUM_NEW_POINTS = 200
NUM_OLD_POINTS = 100

REWARD_MIN = 0.1
REWARD_MAX = 0.9

MEAN = 0
SIGMA = 1  # identity matrix

MIN_START_STATES = 500

ROLLOUT_HORIZON = 50  # max number of timesteps done during random motion


# For our method as well as the baselines, we train a (64, 64) multi-layer perceptron (MLP) Gaussian
# policy with TRPO [36], implemented with rllab [6]. We use a TRPO step-size of 0.01 and a (32, 32)
# MLP baseline.

MLP_SIZE = (64, 64)
MLP_BASELINE_SIZE = (32, 32)
TRPO_STEP_SIZE = 0.01


BATCH_SIZE = 50000  # timesteps


MAXIMUM_HORIZON = 500  # don't know what this means
MAXIMUM_HORIZON_ANT = 2000  # don't know what this means

DISCOUNT_FACTOR = 0.998

# Goal
RADIUS_GOAL_BALL_ROBOT = 0.03  # 0.03m (don't know what the m means)
RADIUS_GOAL_BALL_MAZE = 0.3  # 0.3m (don't know what the m means)
RADIUS_GOAL_BALL_ANT = 0.5  # 0.5m (don't know what the m means)


# K = totalSamplesStates
# Tb = rolloutHorizon

##########################################
#         SAMPLING INITIAL STATES
##########################################

def makeChoice(x):
    return x[np.random.randint(len(x))]


def makeChoices(x, n):
    num = min(len(x), n)
    return [makeChoice(x) for i in range(num)]


def brownian(start, problem, render=False):
    done = False
    steps = 0
    states = []
    _ = problem.env.reset(start)
    while not done and steps < ROLLOUT_HORIZON:
        if render:
            problem.env.render()

        steps += 1
        action = np.random.normal(
            MEAN, SIGMA, problem.actionSize)  # , numAction)
        state, _, done, _ = problem.env.step(action)
        states.append(state)
        if done:  # we don't care about goal done, otherwise will never advance!
            done = False

    return states


def sampleNearby(problem, starts):
    # This part ensures we have MIN_START_STATES in the start array
    while len(starts) < MIN_START_STATES:
        start = makeChoice(starts)
        print(start)
        newStates = brownian(start, problem, render=True)
        starts.extend(newStates)
        print("Added %s states to starts array. New size: %d" % (len(newStates), len(starts)))

    return makeChoices(starts, problem.num_new_starts)


##########################################
#             EVALUATE
##########################################


def rolloutRewards(problem, policy, state, render=True):
    rewards = []
    problem.env.reset(state)
    for _ in range(problem.horizon):
        if render: problem.env.render()
        action, _ = policy.get_action(state)
        next_state, reward, done, _ = problem.env.step(action)

        rewards.append(reward)
        state = next_state

        if done:
            break

    return rewards


def evaluateState(problem, policy, state):
    cumulativeRewards = []
    for _ in range(problem.n_traj):
        cumulativeReward = np.sum(rolloutRewards(problem, policy, state))
        cumulativeRewards.append(cumulativeReward)
    return np.mean(cumulativeRewards)


def evaluateStates(problem, policy, states):
    return {tuple(state): evaluateState(problem, policy, state) for state in states}


def selectStartsWithGoodDifficulty(starts, rewards):
    return [start for start in starts if REWARD_MIN <= rewards[tuple(start)] <= REWARD_MAX]


##########################################
#         CREATE AND TRAIN THE POLICY
##########################################

def getPolicy(problem):
    baseline = LinearFeatureBaseline(env_spec=problem.env.spec)

    policy = GaussianMLPPolicy(
        env_spec=problem.env.spec,
        hidden_sizes=(64, 64),
        # Fix the variance since different goals will require different variances, making this parameter hard to learn.
        learn_std=problem.learn_std,
        adaptive_std=problem.adaptive_std,
        # this is only used if adaptive_std is true!
        std_hidden_sizes=(16, 16),
        output_gain=problem.output_gain,
        init_std=problem.policy_init_std,
    )

    algo = TRPO(
        env=problem.env,
        policy=policy,
        baseline=baseline,
        batch_size=problem.pg_batch_size,
        max_path_length=problem.horizon,
        n_itr=problem.inner_iters,
        step_size=TRPO_STEP_SIZE,
        plot=False,
    )

    return policy, algo


def trainPolicy(problem, starts, algo, policy):
    trpo_paths = algo.train()
    print(trpo_paths)
    with problem.env.set_kill_outside():
        rewards = evaluateStates(problem, policy, states=starts)
    print(rewards)
    return rewards


ITERATIONS = 5


def generateStarts(problem, starts, startsOld):
    with problem.env.set_kill_outside():
        # starts ← SampleNearby(starts, Nnew)
        starts = sampleNearby(problem, starts)

        # starts.append[sample(startsOld , NOld)]
        starts.extend(makeChoices(startsOld, problem.num_old_starts))

    return starts

def training(problem):
    startsOld = [problem.goal]
    starts = [problem.goal]

    policy, algo = getPolicy(problem)

    for _ in range(ITERATIONS):
        
        starts = generateStarts(problem, starts, startsOld)

        # ρ i ← Unif(starts)
        algo.env.update_start_generator(
            UniformListStateGenerator(
                starts, persistence=problem.persistence, with_replacement=problem.with_replacement,
            )
        )

        # rews ← trainPol(ρ i , π i−1 )
        rewards = trainPolicy(problem, starts, algo, policy)

        # starts ← select(starts, rews, R min , R max )
        starts = selectStartsWithGoodDifficulty(starts, rewards)

        # startsOld.append[starts]
        startsOld.extend(starts)

    return policy
