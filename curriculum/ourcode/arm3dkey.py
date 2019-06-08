from curriculum.envs.arm3d.arm3d_key_env import Arm3dKeyEnv
from curriculum.envs.goal_start_env import GoalStartExplorationEnv
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator

from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline


class Arm3DKey:
    def __init__(self):
        self.inner_env = Arm3dKeyEnv(ctrl_cost_coeff=0)
        self.actionSize = self.inner_env.action_space.shape[0]
        
        self.goal = (1.55, 0.4, -3.75, -1.15, 1.81, -2.09, 0.05)
        self.ultimateGoal = (0.0, 0.3, -0.7,  # first point --> hill
             0.0, 0.3, -0.4,  # second point --> top
             -0.15, 0.3, -0.55)

        self.shouldKill = True
        
        fixed_goal_generator = FixedStateGenerator(state=self.ultimateGoal)
        fixed_start_generator = FixedStateGenerator(state=self.goal)

        self.env = GoalStartExplorationEnv(
            env=self.inner_env,
            start_generator=fixed_start_generator,
            obs2start_transform=lambda x: x[:7],
            goal_generator=fixed_goal_generator,
            obs2goal_transform=lambda x: x[-1 * 9:],  # the goal are the last 9 coords
            terminal_eps=0.03,
            distance_metric='L2',
            extend_dist_rew=False,
            inner_weight=0,
            goal_weight=1,
            terminate_env=True,
        )

        self.baseline = GaussianMLPBaseline(env_spec=self.env.spec)

        self.actionSize = self.env.action_space.shape[0]

        self.kill_radius=None
        self.kill_outside=False
        
       
        self.num_new_starts = 600
        self.num_old_starts = 300
        self.horizon = 500
        self.brownianHorizon = 50
        self.outer_iters = 5000
        self.inner_iters = 2
        self.pg_batch_size = 50000
        self.discount = 0.995

        self.n_traj =3  # only for labeling and plotting (for now, later it will have to be equal to persistence!)

        self.output_gain = 0.1
        self.policy_init_std = 1
        self.learn_std = False
        self.adaptive_std = False

        self.initialBrownianHorizon = 10
        
        

        self.persistence = 1
        self.with_replacement = True