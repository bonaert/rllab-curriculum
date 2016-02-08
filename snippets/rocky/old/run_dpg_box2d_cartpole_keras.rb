require_relative './utils'

seed = 1

params = {
  mdp: {
    _name: "box2d.cartpole_mdp",
  },
  normalize_mdp: true,
  qf: {
    _name: "continuous_nn_keras_q_function",
    bn: true,
  },
  policy: {
    _name: "mean_nn_keras_policy",
    hidden_sizes: [100, 100],#32, 32],
    output_nl: 'tanh',
    bn: true,
  },
  exp_name: "dpg_box2d_cartpole",
  algo: {
    _name: "dpg",
    batch_size: 64,
    n_epochs: 100,
    epoch_length: 1000,
    min_pool_size: 10000,
    replay_pool_size: 100000,
    discount: 0.99,
    qf_weight_decay: 0,
    max_path_length: 100,
    eval_samples: 10000,
    eval_whole_paths: true,
    renormalize_interval: 1000,
    qf_learning_rate: 1e-4,
    policy_learning_rate: 1e-6,
    # normalize_qval: false,
  },
  es: {
    _name: "ou_strategy",
  },
  #n_parallel: 1,
  #snapshot_mode: "none",
  seed: seed,
}
command = to_command(params)
puts command
system(command)
