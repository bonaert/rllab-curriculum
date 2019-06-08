from curriculum.envs.maze.maze_ant.ant_maze_start_env import AntMazeEnv
from curriculum.envs.goal_start_env import GoalStartExplorationEnv
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator

from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline


class AntMaze:
    def __init__(self):
        self.inner_env = AntMazeEnv()
        self.actionSize = self.inner_env.action_space.shape[0]
        
        self.goal = (0, 4)
        self.ultimateGoal = (0, 4)

        
        fixed_goal_generator = FixedStateGenerator(state=self.ultimateGoal)
        fixed_start_generator = FixedStateGenerator(state=self.goal)




        self.env = GoalStartExplorationEnv(
            env=self.inner_env,
            start_generator=fixed_start_generator,
            obs2start_transform=lambda x: x[:15],
            goal_generator=fixed_goal_generator,
            obs2goal_transform=lambda x: x[-3:-1],  
            #terminal_eps=0.03,
            terminal_eps=1.0, #new
            distance_metric='L2',
            extend_dist_rew=False,
            inner_weight=0,
            goal_weight=1,
            terminate_env=True,
        )

        self.baseline = GaussianMLPBaseline(env_spec=self.env.spec)

        self.actionSize = self.env.action_space.shape[0]

        #self.kill_radius=None
        #self.kill_outside=False
        
       
        self.num_new_starts = 200
        self.num_old_starts = 100
        self.horizon = 2000
        self.outer_iters = 2000
        self.inner_iters = 5
        self.pg_batch_size = 120000
        self.discount = 0.995

        self.n_traj =3  # only for labeling and plotting (for now, later it will have to be equal to persistence!)

        self.output_gain = 0.1
        self.policy_init_std = 1
        self.learn_std = False
        self.adaptive_std = False

        self.persistence = 1
        self.with_replacement = True