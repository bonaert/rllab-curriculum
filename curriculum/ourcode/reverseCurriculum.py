import pickle
import numpy as np
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator
from curriculum.state.evaluator import convert_label, label_states, evaluate_states, label_states_from_paths
from curriculum.state.utils import StateCollection

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
        print(cumulativeReward)
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
        baseline=problem.baseline,
        batch_size=problem.pg_batch_size,
        max_path_length=problem.horizon,
        n_itr=problem.inner_iters,
        step_size=TRPO_STEP_SIZE,
        plot=False,
    )

    return policy, algo





ITERATIONS = 5


def brownian(start, problem, horizon, render=False):
    with problem.env.set_kill_outside(kill_outside=problem.env.kill_outside, radius=problem.env.kill_radius):
        done = False
        steps = 0
        states = []
        _ = problem.env.reset(start)
        reachedGoal = False
        while not done and steps < horizon:
            if render:
                problem.env.render()
            steps += 1
            action = np.random.uniform(*problem.env.action_space.bounds)
            obs, _, done, _ = problem.env.step(action)
            states.append(problem.env.start_observation)
            if done:  # we don't care about goal done, otherwise will never advance!
                done = False
                reachedGoal = True

    return states, reachedGoal


def sampleNearby(problem, starts=None, horizon=50, subsample = True, size=10000):
    # This part ensures we have MIN_START_STATES in the start array
    if starts is None or len(starts) == 0:
        starts = [problem.env.reset()]

    i = 0
    problem.env.reset(init_state=starts[i % len(starts)])
    states = [problem.env.start_observation]
    
    while len(states) < size:
        start = starts[i % len(starts)]
        newStates, reachedGoal = brownian(start, problem, horizon, render=True)
        states.extend(newStates)
        print("Reached goal: %s" % reachedGoal)
        print("Added %s states to starts array. New size: %d" % (len(newStates), len(states)))
        

        i += 1
        problem.env.reset(init_state=starts[i % len(starts)])
        states.append(problem.env.start_observation)

    if subsample:
        return makeChoices(states, problem.num_new_starts)
    else:
        return states

def generateStarts(problem, starts, horizon):
    with problem.env.set_kill_outside():
        # starts ← SampleNearby(starts, Nnew)
        starts = sampleNearby(problem, starts, horizon, subsample=True)

        # starts.append[sample(startsOld , NOld)]
        
    return starts

def training(problem):
    policy, algo = getPolicy(problem)

    starts = [problem.goal]
    seedStarts = sampleNearby(problem, starts=starts, horizon=10, size=1000)
    
    with open('mlp.pickled', 'wb') as mlpFile, open('results.txt', 'w') as resultsFile:
        all_starts = StateCollection(distance_threshold=0.03)
        brownian_starts = StateCollection(distance_threshold=0)
        for iteration in range(1, problem.outer_iters):
            
            with problem.env.set_kill_outside():
                print("Sampling new starts - Iteration %d" % iteration)
                starts = sampleNearby(problem, seedStarts, problem.horizon, subsample=False)

            brownian_starts.empty()
            brownian_starts.append(starts)
            starts = brownian_starts.sample(size=problem.num_new_starts)

            if iteration > 0 and all_starts.size > 0:
                print("Adding old starts")
                old_starts = all_starts.sample(problem.num_old_starts)
                starts = np.vstack([starts, old_starts])
                


            # ρ i ← Unif(starts)
            algo.env.update_start_generator(
                UniformListStateGenerator(
                    starts, persistence=problem.persistence, with_replacement=problem.with_replacement,
                )
            )

            # rews ← trainPol(ρ i , π i−1 )
            print("Iteration %d - Training the algorithm" % iteration)
            algo.current_itr = 0
            trpo_paths = algo.train(already_init=iteration > 1)

            print("Evaluating the sucess of the algorithm in this iteration %d..." % iteration)
            [starts, labels] = label_states_from_paths(trpo_paths, n_traj=2, key='goal_reached', as_goal=False, env=problem.env)
            start_classes, text_labels = convert_label(labels)
            labels = np.logical_and(labels[:, 0], labels[:, 1]).astype(int).reshape((-1, 1))
            
            successes = sum([1 for label in labels if label[0] == 1])
            total = len(starts)
            print("Successes: %d / %d" % (successes, total))


            print("Saving weights. Iteration %d" % iteration)
            pickle.dump(policy, mlpFile)

            print("Saving results to file")
            resultsFile.write("Iteration %d - Successes: %d / %d\n" % (iteration, successes, total))
            resultsFile.flush()
            


            #with problem.env.set_kill_outside():
            #    rewards = evaluateStates(problem, policy, states=starts)
            #print(rewards)

            # starts ← select(starts, rews, R min , R max )
            #starts = selectStartsWithGoodDifficulty(starts, rewards)


            filtered_raw_starts = [start for start, label in zip(starts, labels) if label[0] == 1]
            all_starts.append(filtered_raw_starts)
            
            if len(filtered_raw_starts) > 0:
                print("Result: some states with medium difficulty - use those ones as seed start")
                seedStarts = filtered_raw_starts
            elif np.sum(start_classes == 0) > np.sum(start_classes == 1):  # if more low reward than high reward
                print("Result: always low reward: resample 300 states as seed starts")
                seedStarts = all_starts.sample(300)  # sample them from the replay
            else:  # add a tone of noise if all the states I had ended up being high_reward!
                print("Result: always high reward: find new seed starts in 5000 step rollouts")
                with algo.env.set_kill_outside(radius=problem.kill_radius):
                    seedStarts = sampleNearby(problem, starts=starts, horizon=int(problem.horizon * 10), subsample=True)
        

            # startsOld.append[starts]
            #startsOld.extend(starts)

    return policy
