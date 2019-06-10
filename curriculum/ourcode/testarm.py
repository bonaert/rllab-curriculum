import numpy as np
import cloudpickle
from curriculum.ourcode.arm3dkey import Arm3DKey
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy


with open('mlp2.pickled', 'rb') as f:
    policy = cloudpickle.load(f)

problem = Arm3DKey()



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


with open('performance.txt', 'w') as f:
    for explorationSteps in range(2000, 20001, 2000):
        for i in range(10):
            done, step = run(explorationSteps, min(5 * explorationSteps, 20000))


            f.write("%s, %s, %s, %s, %s\n" % (explorationSteps, 5 * explorationSteps, i, done, step))
            f.flush()
            print(explorationSteps, min(5 * explorationSteps, 20000), i, done, step)





