import numpy as np
import cloudpickle
from curriculum.ourcode.arm3dkey import Arm3DKey
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy


with open('mlp.pickled', 'rb') as f:
    policy = cloudpickle.load(f)

problem = Arm3DKey()
state, d, e, f = problem.env.step(np.zeros(7))

while True:
    action, rest = policy.get_action(state)
    state, d, e, f = problem.env.step(action)
    problem.env.render()
