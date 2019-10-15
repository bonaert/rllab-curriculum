import numpy as np
from curriculum.ourcode.arm3dkey import Arm3DKey
from curriculum.ourcode.arm3ddisk import Arm3DDisc
from curriculum.ourcode.antmaze import AntMaze

# export PYTHONPATH=/home/greg/software/spark-2.4.0-bin-hadoop2.7/python/lib/pyspark.zip:/home/greg/software/spark-2.4.0-bin-hadoop2.7/python/lib/py4j-0.10.7-src.zip:/home/greg/MEGAsync/aVUB/MA1/MultiAgent/bonaert/rllab-curriculum
# export PYTHONPATH=/home/greg/software/spark-2.4.0-bin-hadoop2.7/python/lib/pyspark.zip:/home/greg/software/spark-2.4.0-bin-hadoop2.7/python/lib/py4j-0.10.7-src.zip:/home/greg/MEGAsync/aVUB/MA1/MultiAgent/bonaert/rllab-curriculum
# export PYTHONPATH=$(pwd)

def renderEnv(problem, reset=False):
    if reset: problem.env.reset(problem.goal)
    
    while True:
        #action = np.random.uniform(*problem.env.action_bounds)
        #problem.env.step(action)
        problem.env.render()


problem = Arm3DKey()
problem.env.reset(problem.goal)

shouldContinue = 'y'
i  = 0
while shouldContinue == 'y':
    action = np.random.uniform(*problem.env.action_bounds)
    state, d, e, f = problem.env.step(action)
    
    if i % 50 == 0:
        problem.env.render()

    i += 1
    if (i + 1) % 3000 == 0:
        shouldContinue = input('should continue (y/n)?:  ')
    print(i)


while True:
	problem.env.render()



