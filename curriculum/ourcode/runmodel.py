import numpy as np
import cloudpickle
from curriculum.ourcode.arm3dkey import Arm3DKey
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy


with open('mlp.pickled', 'rb') as f:
    policy = cloudpickle.load(f)

problem = Arm3DKey()
problem.env.reset(problem.goal)
shouldContinue = 'y'
i  = 0
while shouldContinue == 'y':
    action = np.random.uniform(*problem.env.action_bounds)
    state, d, e, f = problem.env.step(action)
    
    if i % 100 == 0:
        problem.env.render()

    i += 1
    if (i + 1 ) % 5000 == 0:
        shouldContinue = input('should continue (y/n)?:  ')
    print(i)




done = False
while not done:
    action, rest = policy.get_action(state)
    state, done, e, f = problem.env.step(action)
    problem.env.render()
