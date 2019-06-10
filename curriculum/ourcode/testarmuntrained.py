import numpy as np
import cloudpickle
from curriculum.ourcode.arm3dkey import Arm3DKey
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy





problem = Arm3DKey()
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


def resetAndMoveRandomly(steps):
    problem.env.reset(problem.goal)
    for i in range(steps):
        #if i % 20 == 0: problem.env.render()
        action = np.random.uniform(*problem.env.action_bounds)
        state, d, e, f = problem.env.step(action)
    return state

    
def tryToSolve(state, maxSteps):
    step = 0
    done = False
    while not done and step < maxSteps:
        #if step % 20 == 0: problem.env.render()
        action, rest = policy.get_action(state)
        state, done, e, f = problem.env.step(action)
        step += 1

    return done, step

def run(explorationSteps, solveSteps):
    state = resetAndMoveRandomly(explorationSteps)
    return tryToSolve(state, solveSteps)


with open('performanceUntrainted.txt', 'w') as f:
    for explorationSteps in range(2000, 20001, 2000):
        for i in range(10):
            done, step = run(explorationSteps, min(5 * explorationSteps, 20000))


            f.write("%s, %s, %s, %s, %s\n" % (explorationSteps, 5 * explorationSteps, i, done, step))
            f.flush()
            print(explorationSteps, min(5 * explorationSteps, 20000), i, done, step)





