import numpy as np
from curriculum.ourcode.arm3dkey import Arm3DKey
from curriculum.ourcode.arm3ddisk import Arm3DDisc

# export PYTHONPATH=/home/greg/software/spark-2.4.0-bin-hadoop2.7/python/lib/pyspark.zip:/home/greg/software/spark-2.4.0-bin-hadoop2.7/python/lib/py4j-0.10.7-src.zip:/home/greg/MEGAsync/aVUB/MA1/MultiAgent/bonaert/rllab-curriculum
# export PYTHONPATH=/home/greg/software/spark-2.4.0-bin-hadoop2.7/python/lib/pyspark.zip:/home/greg/software/spark-2.4.0-bin-hadoop2.7/python/lib/py4j-0.10.7-src.zip:/home/greg/MEGAsync/aVUB/MA1/MultiAgent/bonaert/rllab-curriculum
# export PYTHONPATH=$(pwd)

#from curriculum.envs.maze.maze_ant.ant_maze_start_env import AntMazeEnv
#env = AntMazeEnv()

#from curriculum.envs.maze.maze_ant.ant_maze_env import AntMazeEnv
#env = AntMazeEnv()

#from curriculum.envs.maze.maze_ant.ant_target_env import AntEnv
#env = AntEnv()

#from curriculum.envs.maze.maze_swim.swim_maze_env import SwimmerMazeEnv
#env = SwimmerMazeEnv()

#from curriculum.envs.maze.point_maze_env import PointMazeEnv
#env = PointMazeEnv()

def renderEnv(problem, reset=False):
    if reset: problem.env.reset(problem.goal)
    
    while True:
        action = np.random.normal(0, 0.1, problem.actionSize)
        problem.env.step(action)
        problem.env.render()

from reverseCurriculum import brownian, sampleNearby, training

problem = Arm3DKey()
#problem = Arm3DDisc()


#states = brownian(problem.goal, problem, render=True)
#print(states)

#states = sampleNearby(problem, [problem.goal])
#print(len(states))
#print(states[:10])


training(problem)

renderEnv(problem)



