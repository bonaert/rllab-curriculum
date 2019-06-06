from curriculum.envs.arm3d.arm3d_disc_env import Arm3dDiscEnv
from curriculum.envs.goal_start_env import GoalStartExplorationEnv
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator


class Arm3DDisc:
    def __init__(self):
        self.inner_env = Arm3dDiscEnv()

        self.goal = (0.387, 1.137, -2.028, -1.744, 2.029, -0.873, 1.55)
        self.ultimateGoal = (0.0, 0.3, -0.7,  # first point --> hill
                             0.0, 0.3, -0.4,  # second point --> top
                            -0.15, 0.3, -0.55) # third point --> side

        fixed_goal_generator = FixedStateGenerator(state=self.ultimateGoal)
        fixed_start_generator = FixedStateGenerator(state=self.goal)

        self.env = GoalStartExplorationEnv(
            env=self.inner_env,
            start_generator=fixed_start_generator,
            obs2start_transform=lambda x: x[:7],
            goal_generator=fixed_goal_generator,
            obs2goal_transform=lambda x: x[-1 * 9:],  # the goal are the last 9 coords
            terminal_eps=0.3,
            distance_metric='L2',
            extend_dist_rew=False,
            inner_weight=0,
            goal_weight=1,
            terminate_env=True,
        )


        self.actionSize = self.env.action_space.shape[0]
        
       
        self.num_new_starts = 600
        self.num_old_starts = 300
        self.horizon = 500
        self.outer_iters = 50
        self.inner_iters = 5
        self.pg_batch_size = 1000
        self.discount = 0.998

        self.n_traj =3  # only for labeling and plotting (for now, later it will have to be equal to persistence!)

        self.output_gain = 0.1
        self.policy_init_std = 1
        self.learn_std = False
        self.adaptive_std = False

        self.persistence = 1
        self.with_replacement = True

