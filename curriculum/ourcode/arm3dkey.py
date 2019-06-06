from curriculum.envs.arm3d.arm3d_key_env import Arm3dKeyEnv


class Arm3DKey:
    def __init__(self):
        self.env = Arm3dKeyEnv(ctrl_cost_coeff=0)
        self.actionSize = self.env.action_space.shape[0]
        
        self.goal = (1.55, 0.4, -3.75, -1.15, 1.81, -2.09, 0.05)
        self.num_new_starts = 600
        self.num_old_starts = 300
        self.horizon = 500
        self.outer_iters = 5000
        self.inner_iters = 2
        self.pg_batch_size = 50000
        self.discount = 0.995

        self.output_gain = 0.1
        self.policy_hidden_sizes = (64, 64)
        self.policy_init_std = 1
        self.learn_std = False
        self.adaptive_std = False