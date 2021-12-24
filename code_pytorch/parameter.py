import argparse


def get_common_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--replay_buffer_size', type=int, default=10000, help='the size of replay buffer')
    parser.add_argument('--batch_size', type=int, default=512, help='the size of batch')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--tau', type=float, default=0.01, help="how depth we exchange the par of the nn")
    parser.add_argument('--n_step', type=int, default=5, help='the number of step for n_step_TD')
    parser.add_argument("--lr_q", type=float, default=5e-4, help="learning rate for adam optimizer")
    parser.add_argument("--lr_a", type=float, default=1e-4, help="learning rate for adam optimizer")
    parser.add_argument("--lr_c", type=float, default=1e-4, help="learning rate for adam optimizer")
    parser.add_argument('--epsilon', type=float, default=0.2, help='the probability of randomly selected action')
    parser.add_argument('--epsilon_min', type=float, default=0.001, help='the min probability of randomly selected action')
    parser.add_argument('--epsilon_delta', type=float, default=0.00001, help='the delta of epsilon')
    parser.add_argument('--var', type=float, default=1., help='the variance of DDPG used to randomly select actions')
    parser.add_argument('--var_delta', type=float, default=0.99999, help='the delta of variance')
    parser.add_argument('--episodes', type=int, default=1000, help='the number of episodes')
    parser.add_argument('--episodes_to_save', type=int, default=100, help='the number of steps to save the model')
    parser.add_argument('--max_steps', type=int, default=1000, help='the max number of step for a episode')
    parser.add_argument('--learn_steps', type=int, default=50, help='the steps to train network')
    parser.add_argument("--num_units_1", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--num_units_2", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--device", type=str, default="cpu", help="use CPU or GPU")
    parser.add_argument("--action_bound", type=int, default=2, help="the bound of action")
    parser.add_argument("--eva_period", type=int, default=5, help="the evaluate time")
 
    args = parser.parse_args()
    return args